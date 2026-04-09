from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import lightning
import numpy as np
import torch
import torch.nn as nn

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler

from ..modules.neural_decoder import NeuralDecoder, NeuralDecoderModule
from ..modules.voxel_grid import LoDVoxelGrid, LoDVoxelGridModule
from ..renders.implicit_renderer import ImplicitRenderingUtils
from ..utils.implicit_wrappers import (AnchorFiltering, CameraWrapper,
                                       NeuralGaussianWrapper)
from .scaffold_gaussian import (ImplicitGaussian, ImplicitModelMixin,
                                ScaffoldGaussianModel, ScaffoldOptimization)


@dataclass
class OctreeOptimization(ScaffoldOptimization):
    progressive: bool = True
    """ progressively optimize anchors from lower lod levels to higher levels """

    coarse_factor: float = 1.5

    coarse_iter: int = 10_000


@dataclass
class OctreeGaussian(ImplicitGaussian):
    voxel_grid: LoDVoxelGrid = field(default_factory=lambda: LoDVoxelGrid())

    optimization: OctreeOptimization = field(default_factory=lambda: OctreeOptimization())

    def instantiate(self, **kwargs):
        return OctreeGaussianModel(self, **kwargs)


class OctreeGaussianModel(GaussianModel, ImplicitModelMixin):
    def __init__(self, config: OctreeGaussian):
        super().__init__()
        self.config = config

        self._names = ("means", "offsets", "scales", "features", "levels", "extra_levels")

    @torch.no_grad()
    def filter_anchors(self, cameras: Cameras, **kwargs) -> AnchorFiltering:
        """
        :returns anchor_mask: [n_cameras, n_anchors], indicating which anchors are visible in each camera
        """
        anchors = self.get_anchors
        pred_levels = self.voxel_grid.predict_level(anchors, cameras) + self.get_extra_levels
        mapped_levels = self.voxel_grid.map_to_int_level(pred_levels, self.activate_level)
        transition_mask = None
        if getattr(mapped_levels, "_progressive_frac", None) is not None:
            transition_mask = self.get_levels == mapped_levels

        anchor_mask = self.get_levels <= mapped_levels
        scales = self.get_scales[:, :3]
        out_mask = anchor_mask.new_zeros(anchor_mask.shape, dtype=torch.bool)
        for cam_idx in range(len(cameras)):
            processed_camera = CameraWrapper.instantiate(cameras[cam_idx : cam_idx + 1])

            _anchor_mask = anchor_mask[cam_idx]
            _means, _scales = anchors[_anchor_mask], scales[_anchor_mask]
            _quats = _means.new_zeros((_means.shape[0], 4))
            _quats[:, 0] = 1.0
            gaussians = NeuralGaussianWrapper(means=_means, scales=_scales, quats=_quats)
            projections = ImplicitRenderingUtils.project_single(processed_camera, gaussians)

            _tmp = _anchor_mask.new_zeros((_means.shape[0],), dtype=torch.bool)
            _tmp[projections.gaussian_ids] = True
            out_mask[cam_idx, _anchor_mask] = _tmp

        return AnchorFiltering(
            anchor_mask=out_mask,
            mapped_levels=mapped_levels,
            transition_mask=transition_mask,
        )

    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        from simple_knn._C import distCUDA2

        cameras: Cameras = kwargs.pop("cameras", None)
        xyz, rgb = xyz[::2], rgb[::2]
        points = torch.from_numpy(xyz).to(cameras[0].device).float()
        self.voxel_grid = self.config.voxel_grid.instantiate()
        self.voxel_grid.setup(points, cameras)

        anchors, levels = self.voxel_grid.voxelize(points)
        mask = self.voxel_grid.weed_out_by_level(anchors, levels, cameras, self.voxel_grid.visibility_threshold)
        anchors, levels = anchors[mask], levels[mask]
        offsets = anchors.new_zeros((anchors.shape[0], self.config.n_offsets, 3))
        dist2 = torch.clamp_min(distCUDA2(anchors.cuda()), 0.0000001)
        scales = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 6)
        features = anchors.new_zeros((anchors.shape[0], self.config.feature_dim))

        property_dict = {
            "means": nn.Parameter(anchors, requires_grad=True),
            "offsets": nn.Parameter(offsets, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "features": nn.Parameter(features, requires_grad=True),
            "levels": nn.Parameter(levels, requires_grad=False),
            "extra_levels": nn.Parameter(torch.zeros((anchors.shape[0],), dtype=torch.float32), requires_grad=False),
        }
        for name, prop in property_dict.items():
            self.set_property(name, prop)

        # setup neural decoder
        self.neural_decoder: NeuralDecoderModule = self.config.neural_decoder.instantiate(
            feature_dim=self.config.feature_dim,
            n_offsets=self.config.n_offsets,
            n_appearance_embeddings=None,
        )
        self.neural_decoder.setup()

    def setup_from_number(self, n, *args, **kwargs):
        self.voxel_grid = self.config.voxel_grid.instantiate()

        anchors = torch.zeros((n, 3), dtype=torch.float32)
        offsets = torch.zeros((n, self.config.n_offsets, 3), dtype=torch.float32)
        scales = torch.zeros((n, 6), dtype=torch.float32)
        features = torch.zeros((n, self.config.feature_dim), dtype=torch.float32)
        levels = torch.zeros((n,), dtype=torch.int32)
        extra_levels = torch.zeros((n,), dtype=torch.float32)
        property_dict = {
            "means": nn.Parameter(anchors, requires_grad=True),
            "offsets": nn.Parameter(offsets, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "features": nn.Parameter(features, requires_grad=True),
            "levels": nn.Parameter(levels, requires_grad=False),
            "extra_levels": nn.Parameter(extra_levels, requires_grad=False),
        }
        for name, prop in property_dict.items():
            self.set_property(name, prop)

        # setup neural decoder
        self.neural_decoder: NeuralDecoderModule = self.config.neural_decoder.instantiate(
            feature_dim=self.config.feature_dim,
            n_offsets=self.config.n_offsets,
            n_appearance_embeddings=None,
        )
        self.neural_decoder.setup()

    def setup_from_tensors(self, tensors, *args, **kwargs):
        pass  # TODO: Implement this method if needed

    def training_setup(self, module: "lightning.LightningModule"):
        if self.config.optimization.offsets_lr_scheduler.max_steps is None:
            self.config.optimization.offsets_lr_scheduler.max_steps = module.trainer.max_steps

        optimizers, schedulers = [], []

        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent
        assert spatial_lr_scale > 0
        config = self.config.optimization
        factory = config.optimizer
        offsets_lr_init = config.offsets_lr_init * spatial_lr_scale
        offsets_optimizer = factory.instantiate(
            [{"params": [self.gaussians["offsets"]], "name": "offsets"}],
            lr=offsets_lr_init,
            eps=1e-15,
        )
        self._add_optimizer_after_backward_hook_if_available(offsets_optimizer, module)
        config.offsets_lr_scheduler.lr_final *= spatial_lr_scale
        offsets_scheduler = config.offsets_lr_scheduler.instantiate().get_scheduler(
            offsets_optimizer,
            offsets_lr_init,
        )
        optimizers.append(offsets_optimizer)
        schedulers.append(offsets_scheduler)

        # constant properties
        l = [
            {"params": self.gaussians["means"], "lr": config.means_lr_init, "name": "means"},
            {"params": self.gaussians["scales"], "lr": config.scales_lr, "name": "scales"},
            {"params": self.gaussians["features"], "lr": config.features_lr, "name": "features"},
        ]
        constant_lr_optimizer = factory.instantiate(l, lr=0.0, eps=1e-15)
        self._add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)
        optimizers.append(constant_lr_optimizer)

        # neural decoders
        decoder_optimizers, decoder_schedulers = self.neural_decoder.training_setup(pl_module=module)
        optimizers += decoder_optimizers
        schedulers += decoder_schedulers

        # setup progressive training
        self._activate_level = self.max_level
        if self.config.optimization.progressive:
            self._activate_level = (
                np.searchsorted(self.coarse_intervals, module.trainer.global_step) + 1 + self.start_level
            )
        module.on_train_batch_end_hooks.append(self.activate_level_update)

        return optimizers, schedulers

    @classmethod
    def activate_level_update(
        cls, outputs, batch, gaussian_model: "OctreeGaussianModel", global_step, pl_module: lightning.LightningModule
    ):
        if gaussian_model.config.optimization.progressive:
            gaussian_model._activate_level = (
                np.searchsorted(gaussian_model.coarse_intervals, global_step) + 1 + gaussian_model.start_level
            )

    def _add_optimizer_after_backward_hook_if_available(self, optimizer, pl_module):
        hook = getattr(optimizer, "on_after_backward", None)
        if hook is None:
            return
        pl_module.on_after_backward_hooks.append(hook)

    @property
    def max_level(self):
        return self.voxel_grid.max_level.item()

    @property
    def start_level(self):
        return self.voxel_grid.start_level.item()

    @property
    def activate_level(self) -> int:
        if getattr(self, "_activate_level", None) is None:
            self._activate_level = self.max_level
        return self._activate_level

    @property
    def coarse_intervals(self):
        if getattr(self, "_coarse_intervals", None) is None:
            self._coarse_intervals = []
            if self.config.optimization.progressive:
                num_level = self.max_level - self.start_level + 1
                if num_level > 0:
                    q = 1.0 / self.config.optimization.coarse_factor
                    a1 = self.config.optimization.coarse_iter * (1 - q) / (1 - q**num_level)
                    temp_interval = 0
                    for i in range(num_level):
                        interval = a1 * q**i + temp_interval
                        temp_interval = interval
                        self._coarse_intervals.append(interval)

        return self._coarse_intervals

    def set_property(self, name: str, value: torch.Tensor):
        if not value.is_floating_point():
            self.gaussians[name] = nn.Parameter(value, requires_grad=False)
        else:
            self.gaussians[name] = value

    def set_properties(self, properties: Dict[str, torch.Tensor]):
        for name in self.property_names:
            self.set_property(name, properties[name])

    def get_property_names(self):
        return self._names
