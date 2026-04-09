from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from gsplat.rasterize_to_weights import rasterize_to_weights
from torch_scatter import scatter_sum
from tqdm import tqdm

from ..utils.implicit_wrappers import NeuralGaussianWrapper, ProjectionWrapper
from .octree_controller import OctreeController, OctreeDensityControllerImpl


@dataclass
class PartitionableDensityController(OctreeController):
    def instantiate(self, *args, **kwargs):
        return PartitionableDensityControllerImpl(self)


class PartitionableDensityControllerImpl(OctreeDensityControllerImpl):
    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)

        pl_module.on_train_start_hooks.append(self._initialize_anchors_on_train_start)

    def _initialize_anchors_on_train_start(self, gaussian_model, pl_module):
        initialize_from = pl_module.hparams.get("initialize_from", None)
        if initialize_from is None or not osp.exists(initialize_from):
            return
        cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
        device = gaussian_model.get_xyz.device
        n_anchors = gaussian_model.get_anchors.shape[0]

        bg_color = torch.zeros((3,), device=device, dtype=torch.float32)
        anchor_weights = torch.zeros((n_anchors,), device=device, dtype=torch.float32)
        for cam_idx, camera in tqdm(enumerate(cameras), desc="Selecting visible anchors", total=len(cameras)):
            camera = camera.to_device(device)
            outputs = pl_module.renderer(camera, gaussian_model, bg_color, render_types=[])
            gaussians: NeuralGaussianWrapper = outputs["neural_gaussians"]
            projections: ProjectionWrapper = outputs["projections"]

            image_width, image_height = int(camera.width), int(camera.height)
            tile_size = getattr(pl_module.renderer.config, "block_size", 16)
            blend_weights = rasterize_to_weights(
                means2d=projections.means2d,
                conics=projections.conics,
                opacities=projections.concatenate_feature(gaussians.camera_ids, gaussians.opacities),
                image_width=image_width,
                image_height=image_height,
                isect_offsets=projections.isect_offsets,
                flatten_ids=projections.flatten_ids,
                pixel_weights=projections.means2d.new_ones((1, image_height, image_width)),
                tile_size=tile_size,
                packed=True,
            )[2].squeeze(0)
            anchor_weights += scatter_sum(blend_weights, projections.anchor_ids, dim_size=n_anchors)

        threshold = torch.quantile(anchor_weights, 0.75)
        mask = anchor_weights > threshold
        optimizers = self._exclude_invalid_optimizers(gaussian_model, pl_module.trainer.optimizers)
        self.prune_anchors(mask, gaussian_model, optimizers)
        self.density_status.prune_buffers(mask)

        cam_centers = cameras.camera_center
        camera_info = torch.cat([cam_centers, torch.ones_like(cam_centers[..., 0:1])], dim=-1).to(self.camera_info)
        delattr(self, "_camera_info")
        self.register_buffer("_camera_info", camera_info)
