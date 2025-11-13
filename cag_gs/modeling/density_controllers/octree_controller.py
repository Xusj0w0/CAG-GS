from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from einops import repeat
from lightning import LightningModule
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from internal.cameras.cameras import Cameras
from internal.density_controllers.density_controller import (
    DensityController, DensityControllerImpl)
from internal.density_controllers.density_controller import \
    Utils as OptimStatManipulator

from ..models.octree_gaussian import OctreeGaussianModel
from ..models.scaffold_gaussian import (NeuralGaussianWrapper,
                                        ScaffoldGaussianModel)
from ..renders.implicit_renderer import ProjectionWrapper
from .scaffold_controller import (DensifyCandidates, ScaffoldController,
                                  ScaffoldDensityControllerImpl)


@dataclass
class OctreeController(ScaffoldController):
    def instantiate(self, *args, **kwargs):
        return OctreeDensityControllerImpl(self, *args, **kwargs)


class OctreeDensityControllerImpl(ScaffoldDensityControllerImpl):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.config: OctreeController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            cameras: Cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
            cam_centers = cameras.camera_center
            camera_info = torch.cat([
                cam_centers, cam_centers.new_ones((cam_centers.shape[0], 1))
            ], dim=-1)  # fmt: skip

            self.register_buffer("_camera_info", camera_info)

    def densify_anchors(self, grads, primitive_reset_mask, gaussian_model: OctreeGaussianModel, optimizers):
        n_anchors_init, n_offsets = gaussian_model.n_anchors, gaussian_model.n_offsets
        grads[~primitive_reset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, n_offsets), dim=-1) / (
            torch.sum(primitive_reset_mask.reshape(-1, n_offsets), dim=-1) + 1e-6
        )
        for cur_level in range(gaussian_model.max_level):
            self._densify_per_level(
                n_anchors_init=n_anchors_init,
                grads=grads,
                anchor_grads=anchor_grads,
                cur_level=cur_level,
                gaussian_model=gaussian_model,
                optimizers=optimizers,
            )

    def _densify_per_level(
        self,
        n_anchors_init: int,
        grads: torch.Tensor,
        anchor_grads: torch.Tensor,
        cur_level: int,
        gaussian_model: OctreeGaussianModel,
        optimizers: List[torch.optim.Optimizer],
    ):
        n_offsets = gaussian_model.n_offsets

        update_value = gaussian_model.config.voxel_grid.fork**self.config.densification_ratio
        levels = gaussian_model.get_levels
        level_mask = levels == cur_level
        level_mask_ds = levels == cur_level + 1
        if torch.sum(level_mask) == 0:
            return
        cur_size = gaussian_model.voxel_grid.voxel_size / (float(gaussian_model.config.voxel_grid.fork) ** cur_level)
        ds_size = cur_size / gaussian_model.config.voxel_grid.fork

        # update threshold
        cur_threshold = self.config.densify_grad_threshold * (update_value**cur_level)
        ds_threshold = cur_threshold * update_value
        extra_threshold = cur_threshold * self.config.extra_ratio
        # mask from grad threshold
        grad_mask = (grads >= cur_threshold) & (grads < ds_threshold)
        grad_mask_ds = grads >= ds_threshold
        grad_mask_extra = anchor_grads >= extra_threshold

        # if prev level add anchors, mask size will dismatch gaussian_model.get_anchors
        n_anchors_diff = gaussian_model.n_anchors - n_anchors_init
        if n_anchors_diff > 0:
            grad_mask = torch.cat([grad_mask, grad_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0)
            grad_mask_ds = torch.cat([grad_mask_ds, grad_mask_ds.new_zeros((n_anchors_diff * n_offsets,))], dim=0)
            grad_mask_extra = torch.cat([grad_mask_extra, grad_mask_extra.new_zeros((n_anchors_diff,))], dim=0)

        # calculate grad mask: grad > thresh and level == current level (next level)
        level_mask_repeat = repeat(level_mask, "n -> (n o)", o=n_offsets)
        grad_mask = torch.logical_and(grad_mask, level_mask_repeat)
        grad_mask_ds = torch.logical_and(grad_mask_ds, level_mask_repeat)
        grad_mask_extra = torch.logical_and(grad_mask_extra, level_mask)

        # if all level are activated, decide whether to update extra_levels by anchor grad
        # in renderer, predicted levels is added by extra_levels
        # means that as the times anchor grad exceed thresh increases, the anchor will be considered more fined during rendering
        if gaussian_model.activate_level >= gaussian_model.max_level:
            gaussian_model.set_property(
                "extra_levels",
                gaussian_model.get_extra_levels + self.config.extra_up * grad_mask_extra.float(),
            )

        primitives = gaussian_model.get_anchors.unsqueeze(dim=1) + (
            gaussian_model.get_offsets * gaussian_model.get_scales[:, :3].unsqueeze(dim=1)
        )
        candidates = self.filter_primitives(
            gaussian_model=gaussian_model,
            primitives=primitives,
            grad_mask=grad_mask,
            res_level=cur_level,
            level_mask=level_mask,
            is_next_level=False,
        )
        candidates_ds = self.filter_primitives(
            gaussian_model=gaussian_model,
            primitives=primitives,
            grad_mask=grad_mask_ds,
            res_level=cur_level + 1,
            level_mask=level_mask_ds,
            is_next_level=True,
        )

        if candidates.n_anchors > 0:
            property_dict = candidates.get_property_dict(
                gaussian_model=gaussian_model, voxel_size=cur_size, scatter_mode=self.config.scatter_mode
            )
            new_properties = OptimStatManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties
        if candidates_ds.n_anchors > 0:
            property_dict = candidates_ds.get_property_dict(
                gaussian_model=gaussian_model, voxel_size=ds_size, scatter_mode=self.config.scatter_mode
            )
            new_properties = OptimStatManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

    def filter_primitives(
        self,
        gaussian_model: OctreeGaussianModel,
        primitives: torch.Tensor,  # Avoid duplicate calculations
        grad_mask: torch.Tensor,
        res_level: torch.Tensor,
        level_mask: torch.Tensor,  # Avoid duplicate calculations
        is_next_level: bool = False,
    ):
        """
        1. primitives are filtered by grad mask (grad_mask)
        2. convert to grids and select unique grids (unique_indices)
        3. filter by existing anchors (if overlap);
        4. filter by predicted level according to distances to train cameras (step 3/4 -> keep_mask)
        """
        voxel_size = gaussian_model.voxel_grid.voxel_size / (float(gaussian_model.config.voxel_grid.fork) ** res_level)

        # filter by grad mask
        candidate_primitives = primitives.view(-1, 3)[grad_mask]

        # convert to grids and select unique grids
        # `unique_indices` is a (candidate_grids.shape[0], ) long tensor
        # same grids are marked with same value
        existing_grids = gaussian_model.voxel_grid.point2grid(gaussian_model.get_anchors[level_mask], voxel_size)
        candidate_grids = gaussian_model.voxel_grid.point2grid(candidate_primitives, voxel_size)
        candidate_grids, unique_indices = torch.unique(candidate_grids, return_inverse=True, dim=0)

        # initial values
        filtered_anchors = candidate_primitives.new_zeros((0, 3))
        filtered_levels = gaussian_model.get_levels.new_zeros((0,))
        keep_mask = existing_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.bool)

        # if is current level, then directly filter by existing anchors and weed out by cameras
        # if is next level, execute filtering after activate_level == max_level
        # and current level shouldn't exceed max_level
        if not is_next_level or (
            gaussian_model.activate_level >= gaussian_model.max_level and res_level < gaussian_model.max_level
        ):
            if candidate_grids.shape[0] > 0:
                # don't filter by existing anchors
                if self.config.overlap < 0:
                    keep_mask = existing_grids.new_ones((candidate_grids.shape[0],), dtype=torch.bool)
                    candidate_anchors = gaussian_model.voxel_grid.grid2point(candidate_grids, voxel_size)
                else:
                    keep_mask = self.filter_exsiting_grids(candidate_grids, existing_grids)
                    candidate_anchors = gaussian_model.voxel_grid.grid2point(candidate_grids[keep_mask], voxel_size)

                candidate_levels = gaussian_model.get_levels.new_ones((candidate_anchors.shape[0],)) * res_level
                keep_mask_cam = gaussian_model.voxel_grid.weed_out_by_level(
                    anchors=candidate_anchors,
                    levels=candidate_levels,
                    cameras=self.camera_info,
                )
                keep_mask[keep_mask.clone()] = keep_mask_cam
                filtered_anchors, filtered_levels = candidate_anchors[keep_mask_cam], candidate_levels[keep_mask_cam]

        return DensifyCandidates(
            anchors=filtered_anchors,
            levels=filtered_levels,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )

    @property
    def camera_info(self) -> torch.Tensor:
        self._camera_info: torch.Tensor
        return self._camera_info

    def on_load_checkpoint(self, module, checkpoint):
        state_dict = {
            k.replace("density_controller.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("density_controller")
        }
        assert "_camera_info" in state_dict, "Camera infos not found in checkpoint."
        self.register_buffer("_camera_info", torch.zeros_like(state_dict["_camera_info"]))
        super().load_state_dict(state_dict)
