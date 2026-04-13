from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from internal.cameras import Cameras

from ..modules.neural_decoder import NeuralDecoderModule
from ..utils.implicit_wrappers import NeuralGaussianWrapper
from .octree_gaussian import OctreeGaussian, OctreeGaussianModel
from .scaffold_gaussian import ImplicitModelMixin


class PartitionableOctreeGaussian(OctreeGaussian):
    def instantiate(self, *args, **kwargs):
        return PartitionableOctreeGaussianModel(self)


class PartitionableOctreeGaussianModel(OctreeGaussianModel):
    config: PartitionableOctreeGaussian

    neural_decoder: List[NeuralDecoderModule]

    def load_state_dict(self, state_dict, strict=True):
        if "_anchor_start_ids_per_block" in state_dict:
            anchor_start_ids_per_block = state_dict["_anchor_start_ids_per_block"]
            self._anchor_start_ids_per_block: torch.Tensor
            self.register_buffer("_anchor_start_ids_per_block", torch.zeros_like(anchor_start_ids_per_block))
            neural_decoder = nn.ModuleList(
                [deepcopy(self.neural_decoder) for _ in range(len(anchor_start_ids_per_block))]
            )
            delattr(self, "neural_decoder")
            self.neural_decoder = neural_decoder
        super().load_state_dict(state_dict, strict)

    @property
    def num_blocks(self):
        return len(self._anchor_start_ids_per_block)

    def generate_neural_gaussians(self, cameras: Cameras):
        # select visible anchors
        anchor_filter = self.filter_anchors(cameras)

        # get valid camera indices and anchor indices
        valid_camera_ids, valid_anchor_ids = torch.nonzero(anchor_filter.anchor_mask, as_tuple=True)
        valid_anchors_per_camera = anchor_filter.anchor_mask.sum(dim=-1)
        num_valid_anchors = len(valid_camera_ids)

        # get per-anchor properties for generating gaussians
        anchors = self.get_anchors[valid_anchor_ids]
        scales = self.get_scales[valid_anchor_ids]
        offsets = self.get_offsets[valid_anchor_ids]
        features = self.compute_features(valid_camera_ids, valid_anchor_ids)

        # encode view
        cam_centers = torch.repeat_interleave(cameras.camera_center, valid_anchors_per_camera, dim=0)
        view_features = self.neural_decoder[0].encode_view(anchors, cam_centers)

        # decode opacities, covariances, colors
        opacities, covariances, colors = [], [], []
        for i in range(self.num_blocks):
            anchor_start_id = self._anchor_start_ids_per_block[i]
            anchor_end_id = self._anchor_start_ids_per_block[i + 1] if i < self.num_blocks - 1 else self.n_anchors
            block_valid_mask = (valid_anchor_ids >= anchor_start_id) & (valid_anchor_ids < anchor_end_id)
            view_features_, features_ = view_features[block_valid_mask], features[block_valid_mask]
            features_ = self.neural_decoder[i].decode_feature_bank(view_features_, features_)
            opacities.append(self.neural_decoder[i].decode_opacity(view_features_, features_).clamp(max=1.0))
            covariances.append(self.neural_decoder[i].decode_covariance(view_features_, features_).reshape(-1, 7))
            colors.append(self.neural_decoder[i].decode_color(view_features_, features_).reshape(-1, 3))
        opacities = anchor_filter.apply_prog(torch.cat(opacities, dim=0)).reshape(-1, 1)
        covariances = torch.cat(covariances, dim=0)
        colors = torch.cat(colors, dim=0)

        # mask invalid gaussians
        valid_gaussian_mask = opacities.squeeze() > 0.0
        concatenated = repeat(torch.cat([anchors, scales], dim=-1), "n c -> (n o) c", o=self.n_offsets)
        concatenated = torch.cat([concatenated, offsets.reshape(-1, 3), covariances, opacities, colors], dim=-1)
        concatenated = concatenated[valid_gaussian_mask]
        anchors, scalings_o, scalings_s, offsets, scales, quats, opacities, colors = torch.split(
            concatenated, [3, 3, 3, 3, 3, 4, 1, colors.shape[-1]], dim=-1
        )
        valid_anchor_ids = repeat(valid_anchor_ids, "n -> (n o)", o=self.n_offsets)[valid_gaussian_mask]
        valid_offset_ids = repeat(torch.arange(self.n_offsets).to(valid_anchor_ids), "o -> (n o)", n=num_valid_anchors)
        valid_offset_ids = valid_offset_ids[valid_gaussian_mask]
        indices_per_camera, idx_start = [0], 0
        for valid_num in valid_anchors_per_camera.tolist():
            st, ed = idx_start * self.n_offsets, (idx_start + valid_num) * self.n_offsets
            indices_per_camera.append(indices_per_camera[-1] + valid_gaussian_mask[st:ed].sum().item())
            idx_start += valid_num

        # compute neural gaussian properties
        means = anchors + offsets * scalings_o
        scales = F.sigmoid(scales) * scalings_s
        quats = self.rotation_activation(quats)
        opacities = opacities.squeeze(-1)

        self._neural_gaussians = NeuralGaussianWrapper(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            colors=colors,
            anchor_ids=valid_anchor_ids,
            offset_ids=valid_offset_ids,
            camera_ids=indices_per_camera,
        )
        return self._neural_gaussians
