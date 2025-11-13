from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
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


@dataclass
class ScaffoldController(DensityController):
    densification: bool = True

    overlap: int = 1
    """maximum number of overlap anchors, <0 for no limit"""

    success_threshold: float = 0.8

    densification_ratio: float = 0.2

    extra_ratio: float = 0.25

    extra_up: float = 0.02

    update_from_iter: int = 500

    densify_from_iter: int = 1_500

    densify_until_iter: int = 15_000

    densification_interval: int = 100

    densify_grad_threshold: float = 2e-4

    cull_opacity_threshold: float = 0.005

    camera_extent_factor: float = 1.0

    scene_extent_override: float = -1.0

    absgrad: bool = False

    scatter_mode: Literal["max", "mean"] = "max"

    def instantiate(self, *args, **kwargs):
        return ScaffoldDensityControllerImpl(self, *args, **kwargs)


class ScaffoldDensityControllerImpl(DensityControllerImpl):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.config: ScaffoldController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            self.cameras_extent = (
                pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor
            )
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor

            if self.config.scene_extent_override > 0:
                self.cameras_extent = self.config.scene_extent_override
                self.prune_extent = self.config.scene_extent_override
                print(f"Override scene extent with {self.config.scene_extent_override}")

            device = pl_module.device
            n_anchors = pl_module.gaussian_model.n_anchors
            n_offsets = pl_module.gaussian_model.n_offsets
            self.density_status = DensityStatus(n_offsets, absgrad=self.config.absgrad)
            self.density_status.init_status(n_anchors, device)

            self.batch_size = pl_module.trainer.datamodule.hparams.get("batch_size", 1)

    def before_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: Union[ScaffoldGaussianModel, OctreeGaussianModel],
        optimizers: List,
        global_step: int,
        pl_module: LightningModule,
    ):
        if global_step >= self.config.densify_until_iter:
            return

        outputs["viewspace_points"].retain_grad()

    def after_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: Union[ScaffoldGaussianModel, OctreeGaussianModel],
        optimizers: List,
        global_step: int,
        pl_module: LightningModule,
    ) -> None:
        if not self.config.densification or global_step >= self.config.densify_until_iter:
            return

        if global_step >= self.config.update_from_iter:
            with torch.no_grad():
                self.density_status.update_status(outputs, gaussian_model.n_anchors)

            if global_step >= self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                property_optimizers = []
                for opt in optimizers:
                    if all([p["name"] in gaussian_model._names for p in opt.param_groups]):
                        property_optimizers.append(opt)
                self.densify_and_prune(gaussian_model, property_optimizers)

    def densify_and_prune(self, gaussian_model: Union[ScaffoldGaussianModel, OctreeGaussianModel], optimizers: list):
        # determine which anchors need to be densified
        grads_norm = self.density_status.primitive_gradient_accum / self.density_status.primitive_denom
        grads_norm[grads_norm.isnan()] = 0.0
        denom_thresh = self.config.densification_interval * self.config.success_threshold * 0.5
        primitive_reset_mask = self.density_status.primitive_denom > denom_thresh

        # densify anchors
        self.densify_anchors(grads_norm, primitive_reset_mask, gaussian_model, optimizers)

        # enlarge buffers to align with densified anchors
        self.density_status.enlarge_buffers(gaussian_model.n_anchors, primitive_reset_mask)

        # determine which anchors need to be pruned
        anchor_denom_thresh = self.config.densification_interval * self.config.success_threshold
        opacity_mask = (
            self.density_status.anchor_opacity_accum
            < self.config.cull_opacity_threshold * self.density_status.anchor_denom
        )
        denom_mask = self.density_status.anchor_denom > anchor_denom_thresh
        keep_mask = ~torch.logical_and(opacity_mask, denom_mask)

        # prune anchors
        self.prune_anchors(keep_mask, gaussian_model, optimizers)

        # prune buffers
        self.density_status.prune_buffers(keep_mask, denom_mask)

        self.density_status.register_status()

    def densify_anchors(self, grads, primitive_reset_mask, gaussian_model: ScaffoldGaussianModel, optimizers):
        grads[primitive_reset_mask] = 0.0
        grad_thresh = self.config.densify_grad_threshold
        grad_mask = grads >= grad_thresh

        primitives = gaussian_model.get_anchors.unsqueeze(1) + (
            gaussian_model.get_offsets * gaussian_model.get_scales[:, :3].unsqueeze(1)
        )
        candidates = self.filter_primitives(
            gaussian_model=gaussian_model,
            primitives=primitives,
            grad_mask=grad_mask,
        )

        if candidates.n_anchors > 0:
            property_dict = candidates.get_property_dict(
                gaussian_model=gaussian_model,
                voxel_size=gaussian_model.voxel_grid.voxel_size,
                scatter_mode=self.config.scatter_mode,
            )
            new_properties = OptimStatManipulator.cat_tensors_to_properties(property_dict, gaussian_model, optimizers)
            gaussian_model.properties = new_properties

    def densify_anchors_paperversion(
        self, grads, primitive_reset_mask, gaussian_model: ScaffoldGaussianModel, optimizers
    ):
        n_anchors_init, n_offsets = gaussian_model.n_anchors, gaussian_model.n_offsets
        for i in range(gaussian_model.config.update_depth):
            cur_thresh = self.config.densify_grad_threshold * ((gaussian_model.config.update_hierachy_factor // 2) ** i)
            grad_mask = grads >= cur_thresh
            grad_mask = torch.logical_and(grad_mask, primitive_reset_mask)

            rand_mask = torch.rand_like(grad_mask.float()).to(grad_mask.device) > (0.5 ** (i + 1))
            grad_mask = torch.logical_and(grad_mask, rand_mask)

            n_anchors_diff = gaussian_model.n_anchors - n_anchors_init
            if n_anchors_diff <= 0:
                if i > 0:
                    continue
            else:
                grad_mask = torch.cat([grad_mask, grad_mask.new_zeros((n_anchors_diff * n_offsets,))], dim=0)

            primitives = gaussian_model.get_anchors.unsqueeze(1) + (
                gaussian_model.get_offsets * gaussian_model.get_scales[:, :3].unsqueeze(1)
            )
            size_factor = gaussian_model.config.update_init_factor // (gaussian_model.config.update_hierachy_factor**i)
            cur_size = gaussian_model.voxel_grid.voxel_size * size_factor
            candidates = self.filter_primitives(
                gaussian_model=gaussian_model,
                primitives=primitives,
                grad_mask=grad_mask,
                voxel_size=cur_size,
            )

            if candidates.n_anchors > 0:
                property_dict = candidates.get_property_dict(
                    gaussian_model=gaussian_model, voxel_size=cur_size, scatter_mode=self.config.scatter_mode
                )
                new_properties = OptimStatManipulator.cat_tensors_to_properties(
                    property_dict, gaussian_model, optimizers
                )
                gaussian_model.properties = new_properties

    def prune_anchors(self, keep_mask, gaussian_model: ScaffoldGaussianModel, optimizers):
        new_properties = OptimStatManipulator.prune_properties(keep_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_properties

    def filter_primitives(
        self,
        gaussian_model: ScaffoldGaussianModel,
        primitives: torch.Tensor,
        grad_mask: torch.Tensor,
        voxel_size: Optional[float] = None,
    ):
        if voxel_size is None:
            voxel_size = gaussian_model.voxel_grid.voxel_size

        # filter by grad mask
        candidate_primitives = primitives.view(-1, 3)[grad_mask]

        # convert to grids and select unique grids
        # `unique_indices` is a (candidate_grids.shape[0], ) long tensor
        # same grids are marked with same value
        existing_grids = gaussian_model.voxel_grid.point2grid(gaussian_model.get_anchors, voxel_size=voxel_size)
        candidate_grids = gaussian_model.voxel_grid.point2grid(candidate_primitives, voxel_size=voxel_size)
        candidate_grids, unique_indices = torch.unique(candidate_grids, return_inverse=True, dim=0)

        # initial values
        filtered_anchors = candidate_primitives.new_zeros((0, 3))
        keep_mask = existing_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.bool)

        if self.config.overlap < 0:
            keep_mask = existing_grids.new_ones((candidate_grids.shape[0],), dtype=torch.bool)
            filtered_anchors = gaussian_model.voxel_grid.grid2point(candidate_grids, voxel_size=voxel_size)
        else:
            keep_mask = self.filter_exsiting_grids(candidate_grids, existing_grids)
            filtered_anchors = gaussian_model.voxel_grid.grid2point(candidate_grids[keep_mask], voxel_size=voxel_size)

        return DensifyCandidates(
            anchors=filtered_anchors,
            grad_mask=grad_mask,
            unique_indices=unique_indices,
            keep_mask=keep_mask,
        )

    def filter_exsiting_grids(
        self,
        candidate_grids: torch.Tensor,
        existing_grids: torch.Tensor,
        max_chunk_size: int = 1 << 15,
    ):
        count = candidate_grids.new_zeros((candidate_grids.shape[0],), dtype=torch.int)
        if max_chunk_size > 0:
            for st in range(0, existing_grids.shape[0], max_chunk_size):
                ed = min(st + max_chunk_size, existing_grids.shape[0])
                cur_existing_grids = existing_grids[st:ed]
                matches = (candidate_grids.unsqueeze(1) == cur_existing_grids).all(-1)
                count += matches.sum(-1)
        else:
            count = (candidate_grids.unsqueeze(1) == cur_existing_grids).all(-1).sum(-1)

        return count < self.config.overlap

    def on_load_checkpoint(self, module, checkpoint):
        state_dict = {
            k.replace("density_controller.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("density_controller")
        }
        super().load_state_dict(state_dict)

    def after_density_changed(self, gaussian_model, optimizers, pl_module):
        self.density_status.register_status()


class DensityStatus(nn.Module):
    def __init__(self, n_offsets: int, absgrad: bool, *args, **kwargs):
        super().__init__()

        self._names = (
            "anchor_opacity_accum",
            "anchor_denom",
            "primitive_gradient_accum",
            "primitive_denom",
        )
        self.n_offsets: int = n_offsets
        self.absgrad: bool = absgrad

    def setup(self, stage: str, pl_module: LightningModule):
        pass

    def init_status(self, n_anchors: int, device):
        self.anchor_opacity_accum = torch.zeros((n_anchors,), device=device, dtype=torch.float32)
        self.anchor_denom = torch.zeros((n_anchors,), device=device, dtype=torch.long)
        self.primitive_gradient_accum = torch.zeros((n_anchors * self.n_offsets,), device=device, dtype=torch.float32)
        self.primitive_denom = torch.zeros((n_anchors * self.n_offsets,), device=device, dtype=torch.long)

        self.register_status()

    def register_status(self):
        for buffer in self._names:
            _buf = getattr(self, buffer)
            delattr(self, buffer)
            self.register_buffer(buffer, _buf)

    def update_status(self, outputs, n_anchors):
        viewspace_point_tensor = outputs["viewspace_points"]
        viewspace_points_grad_scale = outputs["viewspace_points_grad_scale"]
        gaussians: NeuralGaussianWrapper = outputs["neural_gaussians"]
        projections: ProjectionWrapper = outputs["projections"]
        device = gaussians.means.device

        # accumulate opacity
        self.anchor_opacity_accum += scatter_sum(gaussians.opacities, gaussians.anchor_ids, dim=0, dim_size=n_anchors)
        idx = torch.ones_like(gaussians.opacities, dtype=torch.long, device=device)
        self.anchor_denom += scatter_sum(idx, gaussians.anchor_ids, dim=0, dim_size=n_anchors)

        # accumulate gradient
        if self.absgrad:
            xys_grad = viewspace_point_tensor.absgrad
        else:
            xys_grad = viewspace_point_tensor.grad
        if viewspace_points_grad_scale is not None:
            xys_grad = xys_grad * viewspace_points_grad_scale
        grad_norm = torch.norm(xys_grad, dim=-1)
        gaussian_ids = projections.anchor_ids * self.n_offsets + projections.offset_ids
        self.primitive_gradient_accum += scatter_sum(
            grad_norm, gaussian_ids, dim=0, dim_size=n_anchors * self.n_offsets
        )
        idx = torch.ones_like(grad_norm, dtype=torch.long, device=device)
        self.primitive_denom += scatter_sum(idx, gaussian_ids, dim=0, dim_size=n_anchors * self.n_offsets)

    def enlarge_buffers(self, n_anchors: int, primitive_reset_mask: torch.Tensor):
        extra_anchors = n_anchors - self.anchor_denom.shape[0]
        self.anchor_opacity_accum = torch.cat([
            self.anchor_opacity_accum, self.anchor_opacity_accum.new_zeros((extra_anchors,))
        ], dim=0)  # fmt: skip
        self.anchor_denom = torch.cat([
            self.anchor_denom, self.anchor_denom.new_zeros((extra_anchors,))
        ], dim=0)  # fmt: skip

        self.primitive_gradient_accum[primitive_reset_mask] = 0.0
        self.primitive_denom[primitive_reset_mask] = 0
        self.primitive_gradient_accum = torch.cat([
            self.primitive_gradient_accum, self.primitive_gradient_accum.new_zeros((extra_anchors * self.n_offsets,))
        ], dim=0)  # fmt: skip
        self.primitive_denom = torch.cat([
            self.primitive_denom, self.primitive_denom.new_zeros((extra_anchors * self.n_offsets,))
        ], dim=0)  # fmt: skip

        torch.cuda.empty_cache()

    def prune_buffers(self, keep_mask: torch.Tensor, anchor_reset_mask: torch.Tensor):
        self.primitive_gradient_accum = self.primitive_gradient_accum.view(-1, self.n_offsets)[keep_mask].view(-1)
        self.primitive_denom = self.primitive_denom.view(-1, self.n_offsets)[keep_mask].view(-1)
        if anchor_reset_mask.sum() > 0:
            self.anchor_opacity_accum[anchor_reset_mask] = 0.0
            self.anchor_denom[anchor_reset_mask] = 0
        self.anchor_opacity_accum = self.anchor_opacity_accum[keep_mask]
        self.anchor_denom = self.anchor_denom[keep_mask]

        torch.cuda.empty_cache()

    def load_state_dict(self, state_dict, strict: bool = True):
        n_anchors = state_dict["anchor_denom"].shape[0]
        n_offsets = state_dict["primitive_denom"].shape[0] // n_anchors
        self.n_offsets = n_offsets
        self.init_status(n_anchors, device="cpu")
        super().load_state_dict(state_dict, strict)


@dataclass
class DensifyCandidates:
    anchors: Optional[torch.Tensor] = None
    """filtered anchors"""

    levels: Optional[torch.Tensor] = None
    """filtered levels"""

    grad_mask: Optional[torch.Tensor] = None
    """filter primivites by grad_mask"""

    unique_indices: Optional[torch.Tensor] = None
    """convert to grids and select unique grids"""

    keep_mask: Optional[torch.Tensor] = None
    """remove existing anchors & filter by predicted level"""

    @property
    def n_anchors(self) -> int:
        return self.anchors.shape[0]

    def get_property_dict(
        self,
        gaussian_model: Union[ScaffoldGaussianModel, OctreeGaussianModel],
        voxel_size: float,
        scatter_mode: Literal["max", "mean"] = "max",
    ):
        anchors = self.anchors
        scales = gaussian_model.scale_inverse_activation(
            gaussian_model.get_scaling.new_ones((self.n_anchors, 6)) * voxel_size
        )
        offsets = gaussian_model.get_anchors.new_zeros((self.n_anchors, gaussian_model.n_offsets, 3))

        # features
        keep_indices = torch.nonzero(self.keep_mask, as_tuple=True)[0]
        is_keep = self.unique_indices.unsqueeze(1) == keep_indices.unsqueeze(0)
        # avoid OOM
        keep_mask, keep_idx_mapping = self.chunked_nonzero(is_keep)
        indices = (torch.nonzero(self.grad_mask, as_tuple=True)[0] / gaussian_model.n_offsets).long()[keep_mask]
        features = gaussian_model.get_features[indices]
        feature_dim = features.shape[-1]
        if scatter_mode == "max":
            features = scatter_max(features, keep_idx_mapping.unsqueeze(1).expand(-1, feature_dim), dim=0)[0]
        elif scatter_mode == "mean":
            features = scatter_mean(features, keep_idx_mapping.unsqueeze(1).expand(-1, feature_dim), dim=0)
        else:
            raise ValueError(f"scatter_mode {scatter_mode} not supported")

        outputs = {"means": anchors, "scales": scales, "offsets": offsets, "features": features}

        if gaussian_model.is_lod:
            outputs.update({
                "levels": self.levels,
                "extra_levels": gaussian_model.get_extra_levels.new_zeros((self.n_anchors,)),
            })  # fmt: skip
        return outputs

    @classmethod
    def chunked_nonzero(cls, is_keep: torch.Tensor, max_chunk_size: int = 1 << 30):
        rows, cols = is_keep.shape
        rows_per_chunk = max_chunk_size // cols
        keep_mask_list, keep_idx_mapping_list = [], []
        for st in range(0, rows, rows_per_chunk):
            ed = min(st + rows_per_chunk, rows)
            chunk = is_keep[st:ed]
            r, c = torch.nonzero(chunk, as_tuple=True)
            r = r + st

            keep_mask_list.append(r)
            keep_idx_mapping_list.append(c)
        return torch.cat(keep_mask_list, dim=0), torch.cat(keep_idx_mapping_list, dim=0)
