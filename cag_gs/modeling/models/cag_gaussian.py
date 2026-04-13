from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn

from internal.cameras.cameras import Camera
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler

from ..modules.feature_adapter import FeatureAdapter
from ..modules.hash_grid import HashGrid
from ..modules.neural_decoder import NeuralDecoder
from ..modules.voxel_grid import LoDVoxelGrid
from .octree_gaussian import (OctreeGaussian, OctreeGaussianModel,
                              OctreeOptimization)


@dataclass
class ConsistentAnchorGuidedGaussian(OctreeGaussian):
    hash_grid: HashGrid = field(default_factory=lambda: HashGrid())

    feature_adapter: FeatureAdapter = field(default_factory=lambda: FeatureAdapter())

    voxel_grid: LoDVoxelGrid = field(default_factory=lambda: LoDVoxelGrid())

    neural_decoder: NeuralDecoder = field(default_factory=lambda: NeuralDecoder())

    optimization: OctreeOptimization = field(default_factory=lambda: OctreeOptimization())

    def instantiate(self, *args, **kwargs):
        return ConsistentAnchorGuidedGaussianModel(self, *args, **kwargs)


class ConsistentAnchorGuidedGaussianModel(OctreeGaussianModel):
    config: ConsistentAnchorGuidedGaussian

    def compute_features(self, valid_camera_ids, valid_anchor_ids):
        anchor_features = self.get_features[valid_anchor_ids]
        with torch.no_grad():
            hash_features = self._compute_hash_features(valid_camera_ids, valid_anchor_ids)
        return anchor_features + self.feature_adapter(hash_features)

    def get_features_for_rasterization(self, anchor_ids):
        return self._compute_hash_features(None, anchor_ids)

    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        super().setup_from_pcd(xyz, rgb, *args, **kwargs)

        pl_module = kwargs["pl_module"]
        semantic_feature_dim = pl_module.trainer.datamodule.dataparser_outputs.semantic_feature_dim
        self.config.hash_grid.mlp_out_dim = semantic_feature_dim
        self.hash_grid = self.config.hash_grid.instantiate()
        self.hash_grid.setup(stage="fit")

        self.config.feature_adapter.in_dim = self.config.hash_grid.mlp_out_dim
        self.config.feature_adapter.out_dim = self.config.feature_dim
        self.feature_adapter = self.config.feature_adapter.instantiate()
        self.feature_adapter.setup(stage="fit")

    def load_state_dict(self, state_dict, strict=True):
        device = self.gaussians["means"].device

        self.config.hash_grid.mlp_out_dim = state_dict["feature_adapter.weight"].shape[0]
        self.hash_grid = self.config.hash_grid.instantiate()
        self.hash_grid.setup(stage="fit")
        self.hash_grid.to(device)

        self.config.feature_adapter.in_dim = self.hash_grid.config.mlp_out_dim
        self.config.feature_adapter.out_dim = self.config.feature_dim
        self.feature_adapter = self.config.feature_adapter.instantiate()
        self.feature_adapter.setup(stage="fit")
        self.feature_adapter.to(device)
        super().load_state_dict(state_dict, strict)

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)
        hash_optimizers, hash_schedulers = self.hash_grid.training_setup(module)
        adapter_optimizers, adapter_schedulers = self.feature_adapter.training_setup(module)
        return optimizers + hash_optimizers + adapter_optimizers, schedulers + hash_schedulers + adapter_schedulers

    def _compute_hash_features(self, valid_camera_ids, valid_anchor_ids):
        anchors = self.get_anchors[valid_anchor_ids]
        normalized = self._normalize_xyz(anchors)
        mask = ((normalized >= 0.0) & (normalized <= 1.0)).all(dim=-1, keepdim=True)
        normalized = normalized * mask
        hash_features = self.hash_grid(normalized)
        return hash_features

    def _normalize_xyz(self, points: torch.Tensor):
        bbox_min = self.voxel_grid.bounding_box[:3].to(points)
        bbox_max = self.voxel_grid.bounding_box[3:].to(points)
        transformed = self.voxel_grid.apply_transform(points)
        normalized = (transformed - bbox_min) / (bbox_max - bbox_min)
        return normalized
