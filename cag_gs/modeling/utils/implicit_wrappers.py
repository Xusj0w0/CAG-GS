from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from internal.cameras.cameras import Cameras


@dataclass
class AnchorFiltering:
    anchor_mask: torch.Tensor
    mapped_levels: Optional[torch.Tensor] = None
    transition_mask: Optional[torch.Tensor] = None

    def apply_prog(self, opacities: torch.Tensor):
        if self.mapped_levels is not None and self.mapped_levels._progressive_frac is not None:
            frac = self.mapped_levels._progressive_frac[self.anchor_mask]
            transition = self.transition_mask[self.anchor_mask]
            frac[~transition] = 1.0
            return opacities * frac
        return opacities


@dataclass
class CameraWrapper:
    viewmats: torch.Tensor
    Ks: torch.Tensor  # [n_cameras, 4, 4]
    img_width: int
    img_height: int

    @classmethod
    def instantiate(cls, cameras: Cameras):
        assert torch.all(cameras.width == cameras.width[0]) and torch.all(
            cameras.height == cameras.height[0]
        ), "All cameras must have the same width and height."
        viewmats = cameras.world_to_camera.transpose(-1, -2)
        Ks = viewmats.new_zeros((viewmats.shape[0], 3, 3))
        Ks[:, 0, 0] = cameras.fx
        Ks[:, 1, 1] = cameras.fy
        Ks[:, 0, 2] = cameras.cx
        Ks[:, 1, 2] = cameras.cy

        img_width = int(cameras.width[0].item())
        img_height = int(cameras.height[0].item())

        return CameraWrapper(viewmats=viewmats, Ks=Ks, img_width=img_width, img_height=img_height)


@dataclass
class NeuralGaussianWrapper:
    means: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    quats: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None

    anchor_ids: Optional[torch.Tensor] = None
    offset_ids: Optional[torch.Tensor] = None

    camera_ids: Optional[list] = None

    @property
    def n_gaussians(self) -> int:
        try:
            return self.means.shape[0]
        except:
            return 0

    def filter(self, mask: Union[torch.Tensor, slice]) -> "NeuralGaussianWrapper":
        params = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k, None)
            if isinstance(v, torch.Tensor):
                params[k] = v[mask]
            else:
                params[k] = None
        return self.__class__(**params)


@dataclass
class ProjectionWrapper:
    radii: Optional[torch.Tensor] = None
    means2d: Optional[torch.Tensor] = None
    depths: Optional[torch.Tensor] = None
    conics: Optional[torch.Tensor] = None
    compensations: Optional[torch.Tensor] = None

    gaussian_ids: Optional[torch.Tensor] = None
    anchor_ids: Optional[torch.Tensor] = None
    offset_ids: Optional[torch.Tensor] = None

    tiles_per_gauss: torch.Tensor = field(init=False)  # [nnz]
    isect_ids: torch.Tensor = field(init=False)  # [n_isects]
    flatten_ids: torch.Tensor = field(init=False)  # [n_isects]
    isect_offsets: torch.Tensor = field(init=False)  # [C, tile_height, tile_width]

    camera_ids: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self):
        if self.radii is not None and len(self.radii) > 0:
            self.camera_ids.append(len(self.radii))

    def append(
        self,
        other: "ProjectionWrapper",
        gaussian_id_offset: int,
        gaussian2d_id_offset: int,
        isect_id_offset: int,
    ):
        if self.radii is None:
            for k in self.__dataclass_fields__:
                setattr(self, k, getattr(other, k))
            return

        self.radii = torch.cat([self.radii, other.radii], dim=0)
        self.means2d = torch.cat([self.means2d, other.means2d], dim=0)
        self.depths = torch.cat([self.depths, other.depths], dim=0)
        self.conics = torch.cat([self.conics, other.conics], dim=0)
        if self.compensations is not None:
            self.compensations = torch.cat([self.compensations, other.compensations], dim=0)

        self.gaussian_ids = torch.cat([
            self.gaussian_ids, other.gaussian_ids + gaussian_id_offset
        ], dim=0) # fmt: skip
        self.anchor_ids = torch.cat([self.anchor_ids, other.anchor_ids], dim=0)
        self.offset_ids = torch.cat([self.offset_ids, other.offset_ids], dim=0)

        self.tiles_per_gauss = torch.cat([self.tiles_per_gauss, other.tiles_per_gauss], dim=0)
        self.isect_ids = torch.cat([self.isect_ids, other.isect_ids], dim=0)
        self.flatten_ids = torch.cat([self.flatten_ids, other.flatten_ids + gaussian2d_id_offset], dim=0)
        self.isect_offsets = torch.cat([self.isect_offsets, other.isect_offsets + isect_id_offset], dim=0)

        self.camera_ids += [self.camera_ids[-1] + other.camera_ids[-1]]

        return self

    def concatenate_feature(self, camera_ids: list, features: torch.Tensor):
        gaussian_ids, gaussian_id_offset = self.gaussian_ids, 0
        all_features = []
        for cid in range(len(camera_ids) - 1):
            features_cam = features[camera_ids[cid] : camera_ids[cid + 1]]
            num_gaussians2d = self.camera_ids[cid + 1] - self.camera_ids[cid]

            _gaussian_ids, gaussian_ids = gaussian_ids[:num_gaussians2d], gaussian_ids[num_gaussians2d:]
            _gaussian_ids = _gaussian_ids - gaussian_id_offset
            gaussian_id_offset += features_cam.shape[0]

            all_features.append(features_cam[_gaussian_ids])

        return torch.cat(all_features, dim=0)
