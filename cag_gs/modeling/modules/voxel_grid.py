import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from internal.cameras.cameras import Camera, Cameras

__all__ = ["VoxelGrid", "VoxelGridModule", "LoDVoxelGrid", "LoDVoxelGridModule"]


@dataclass
class VoxelGrid:
    default_voxel_size: float = -1.0
    """ voxel size of the grid. If <0, use median 1-NN distance of points as voxel size """

    outlier_ratio: float = 0.001
    """ outlier ratio for building bounding box, use `torch.quantile` to filter outlier points """

    extend_ratio: float = 0.1
    """ extend ratio for building bounding box """

    transform: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 6)
    """ transform of the grid, in the format of [qw, qx, qy, qz, tx, ty, tz] """

    def instantiate(self, points: Optional[torch.Tensor] = None, **kwargs):
        return VoxelGridModule(self, points=points, **kwargs)


class VoxelGridModule(nn.Module):
    def __init__(self, config: VoxelGrid, points: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self.config = config
        self._buffer_names = (
            "transform",
            "bounding_box",
            "voxel_size",
        )

        # setup grid
        self.transform = torch.tensor([1.0] + [0.0] * 6, dtype=torch.float32)
        self.bounding_box = torch.zeros((6,), dtype=torch.float32)
        self.voxel_size = torch.tensor(0.0, dtype=torch.float32)

        if points is not None:
            self.transform = self.get_transform()

            points = points.cuda()  # accelerate setting up
            self.bounding_box = self.get_bounding_box(points)
            self.voxel_size = self.get_voxel_size(points)

        # register grid info
        self.register_grid_info()

    def get_transform(self) -> torch.Tensor:
        return torch.tensor(self.config.transform, dtype=torch.float32)

    def get_bounding_box(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get the bounding box of the grid based on the points.

        :param: [N, 3] tensor of points
        :return: (min_point, max_point) where min_point and max_point are [3] tensors
        """
        if points.shape[0] == 0:
            raise ValueError("No points provided to get bounding box.")

        transformed = self.apply_transform(points)
        min_point = torch.quantile(transformed, self.config.outlier_ratio, dim=0)
        max_point = torch.quantile(transformed, 1 - self.config.outlier_ratio, dim=0)

        # Extend the bounding box
        min_point_ = min_point - (max_point - min_point) * self.config.extend_ratio
        max_point_ = max_point + (max_point - min_point) * self.config.extend_ratio
        return torch.tensor(min_point_.tolist() + max_point_.tolist(), dtype=torch.float32)

    def get_voxel_size(self, points) -> torch.Tensor:
        voxel_size = self.config.default_voxel_size
        if voxel_size < 0:
            from simple_knn._C import distCUDA2

            dist = distCUDA2(points.clone().cuda())
            median_dist = torch.median(dist)
            voxel_size = median_dist.item()
        return torch.tensor(voxel_size, dtype=torch.float32)

    def voxelize(self, points: torch.Tensor, voxel_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        grid_points = self.point2grid(points, voxel_size=voxel_size)
        grid_points = torch.unique(grid_points, dim=0)
        output_points = self.grid2point(grid_points, voxel_size=voxel_size)
        return output_points

    def register_grid_info(self):
        for buffer in self._buffer_names:
            _buf = getattr(self, buffer)
            delattr(self, buffer)
            self.register_buffer(buffer, _buf)

    def point2grid(self, points: torch.Tensor, voxel_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        if voxel_size is None:
            voxel_size = self.voxel_size
        transformed = self.apply_transform(points)
        return torch.round(transformed / voxel_size.to(points)).long()

    def grid2point(self, grid_points: torch.Tensor, voxel_size: Optional[torch.Tensor] = None) -> torch.Tensor:
        if voxel_size is None:
            voxel_size = self.voxel_size
        grid_points = grid_points.float()
        points = grid_points * voxel_size.to(grid_points)
        return self.apply_inverse_transform(points)

    def apply_transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to the points.

        :param points: [N, 3] tensor of points
        :return transformed: transformed points as [N, 3] tensor
        """
        if points.shape[1] != 3:
            raise ValueError("Points must be of shape [N, 3]")
        if self.rotation is None:
            return points
        return points @ self.rotation.T.to(points) + self.transition.to(points)

    def apply_inverse_transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse transformation to the points.

        :param points: [N, 3] tensor of points
        :return transformed: transformed points as [N, 3] tensor
        """
        if points.shape[1] != 3:
            raise ValueError("Points must be of shape [N, 3]")
        if self.rotation is None:
            return points
        return (points - self.transition.to(points)) @ self.rotation.to(points)

    @property
    def rotation(self) -> Optional[torch.Tensor]:
        "return rotation matrix as tensor"
        if not hasattr(self, "_rotation"):
            unit_quat = self.transform.new_zeros((4,))
            unit_quat[0] = 1.0
            if torch.allclose(self.transform[:4], unit_quat):
                self._rotation = None
            else:
                quat = self.transform[:4].clone().detach().to(dtype=torch.float32)
                self._rotation = quaternion_to_matrix(quat)
        return self._rotation

    @property
    def transition(self) -> torch.Tensor:
        "return transition vector as tensor"
        if not hasattr(self, "_transition"):
            self._transition = self.transform[4:].clone().detach().to(dtype=torch.float32)
        return self._transition


@dataclass
class LoDVoxelGrid(VoxelGrid):
    fork: int = 2

    base_layer: int = 11
    """ determine `voxel_size` if not provided: max(box_max - box_min) / (2 ** (base_layer - 1)) """

    max_level: int = -1

    start_level: int = -1

    dist_ratio: float = 0.001
    """ filter distances between camera centers and points use `torch.quantile` """

    level_mapping_mode: Literal["floor", "round", "ceil", "progressive"] = "floor"

    visibility_threshold: float = 0.01

    def instantiate(
        self,
        points: Optional[torch.Tensor] = None,
        cameras: Optional[Cameras] = None,
        **kwargs,
    ):
        return LoDVoxelGridModule(self, points=points, cameras=cameras, **kwargs)


class LoDVoxelGridModule(VoxelGridModule):
    def __init__(
        self,
        config: LoDVoxelGrid,
        points: Optional[torch.Tensor] = None,
        cameras: Optional[Cameras] = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.config = config
        self._buffer_names = (
            "transform",
            "bounding_box",
            "voxel_size",
            "max_level",
            "start_level",
            "standard_dist",
            "visibility_threshold",
        )

        self.transform = torch.tensor([1.0] + [0.0] * 6, dtype=torch.float32)
        self.bounding_box = torch.zeros((6,), dtype=torch.float32)
        self.voxel_size = torch.tensor(0.0, dtype=torch.float32)
        self.max_level = torch.tensor(0, dtype=torch.int32)
        self.start_level = torch.tensor(0, dtype=torch.int32)
        self.standard_dist = torch.tensor(0.0, dtype=torch.float32)
        self.visibility_threshold = torch.tensor(0.0, dtype=torch.float32)

        if points is not None and cameras is not None:
            self.transform = self.get_transform()

            points = points.cuda()
            self.bounding_box = self.get_bounding_box(points)
            self.voxel_size = self.get_voxel_size()

            # get params for LoD prediction
            self.max_level, self.standard_dist = self.get_lod_params(points, cameras)
            start_level = self.config.start_level
            if start_level < 0:
                start_level = int(self.max_level.item() // 2)
            self.start_level = torch.tensor(start_level, dtype=torch.int32)

            visibility_threshold = torch.tensor(self.config.visibility_threshold)
            if visibility_threshold < 0.0:
                anchors, levels = self.voxelize(points)
                mask = self.weed_out_by_level(anchors, levels, cameras, 0.0)
                visibility_threshold = torch.mean(mask.float())
            self.visibility_threshold = visibility_threshold

        # register grid info
        self.register_grid_info()

    def get_voxel_size(self) -> torch.Tensor:
        voxel_size = self.config.default_voxel_size
        if voxel_size < 0:
            if self.bounding_box.shape[0] == 0:
                raise ValueError("Bounding box is not set, cannot determine voxel size.")
            max_dim = torch.max(self.bounding_box[3:] - self.bounding_box[:3]).item()
            voxel_size = max_dim / (2**self.config.base_layer)
        return torch.tensor(voxel_size, dtype=torch.float32)

    def get_lod_params(
        self,
        points: torch.Tensor,
        cameras: Cameras,
        max_chunk_size: int = 1 << 24,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute `max_level` and `standard_dist`, which will be used to predict anchor's lod level based on its distance to camera
        """
        camera_centers = cameras.camera_center.to(device=points.device)
        camera_infos = torch.cat([camera_centers, camera_centers.new_ones((camera_centers.shape[0], 1))], dim=-1)
        num_points, num_cameras = points.shape[0], camera_infos.shape[0]
        if max_chunk_size > 0:  # chunking
            num_per_chunk = self.max_power_of_2(max_chunk_size // num_points)
            dist_min = points.new_zeros((num_cameras,), dtype=torch.float32)
            dist_max = points.new_zeros((num_cameras,), dtype=torch.float32)
            for st in range(0, num_cameras, num_per_chunk):
                ed = min(st + num_per_chunk, num_cameras)
                camera_chunk = camera_infos[st:ed]
                dist = torch.cdist(points, camera_chunk[:, :3], p=2) * camera_chunk[:, -1]
                dist_min[st:ed].copy_(torch.quantile(dist, self.config.dist_ratio, dim=0))
                dist_max[st:ed].copy_(torch.quantile(dist, 1 - self.config.dist_ratio, dim=0))
        else:
            dist = torch.cdist(points, camera_infos[:, :3], p=2) * camera_infos[:, -1]
            dist_min = torch.quantile(dist, self.config.dist_ratio, dim=0)
            dist_max = torch.quantile(dist, 1 - self.config.dist_ratio, dim=0)
        dist_min = torch.quantile(dist_min, self.config.dist_ratio)
        dist_max = torch.quantile(dist_max, 1 - self.config.dist_ratio)

        max_level = torch.tensor(self.config.max_level, dtype=torch.int32)
        if max_level < 0:
            max_level = torch.round(torch.log2(dist_max / dist_min) / math.log2(float(self.config.fork))).int() + 1
        return max_level, dist_max

    def predict_level(self, points: torch.Tensor, cameras: Union[Cameras, torch.Tensor]) -> torch.Tensor:
        if isinstance(cameras, Cameras):
            camera_centers = cameras.camera_center.to(points)
        else:
            camera_centers = cameras[:, :3].to(points)
        dist = torch.cdist(camera_centers, points, p=2)
        pred_level = torch.log2(self.standard_dist / dist) / math.log2(float(self.config.fork))
        return pred_level

    def map_to_int_level(self, levels: torch.Tensor, max_level: int) -> torch.Tensor:
        if self.config.level_mapping_mode == "floor":
            # return MappedLevel(torch.floor(levels).int().clamp(0, max_level))
            mapped_level = torch.floor(levels).int().clamp(0, max_level)
            mapped_level._progressive_frac = None
            return mapped_level
        elif self.config.level_mapping_mode == "round":
            # return MappedLevel(torch.round(levels).int().clamp(0, max_level))
            mapped_level = torch.round(levels).int().clamp(0, max_level)
            mapped_level._progressive_frac = None
            return mapped_level
        elif self.config.level_mapping_mode == "ceil":
            # return MappedLevel(torch.ceil(levels).int().clamp(0, max_level))
            mapped_level = torch.ceil(levels).int().clamp(0, max_level)
            mapped_level._progressive_frac = None
            return mapped_level
        elif self.config.level_mapping_mode == "progressive":
            eps = 1e-4
            pred_level = (levels + 1.0).clamp(1.0 - eps, max_level - eps)
            # mapped_level = MappedLevel(torch.floor(pred_level).int())
            # mapped_level.set_frac(torch.frac(pred_level))
            mapped_level = torch.floor(pred_level).int()
            mapped_level._progressive_frac = torch.frac(pred_level)
            return mapped_level
        else:
            raise ValueError(f"Unknown int level mode: {self.config.level_mapping_mode}")

    def voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_points = points.new_empty((0, 3), dtype=torch.float32)
        all_levels = points.new_empty((0,), dtype=torch.int32)
        for cur_level in range(self.max_level.item()):
            cur_voxel_size = self.voxel_size * (self.config.fork**-cur_level)

            _points = super().voxelize(points, voxel_size=cur_voxel_size)
            _levels = torch.full((_points.shape[0],), cur_level, dtype=torch.int32, device=_points.device)
            all_points = torch.cat([all_points, _points], dim=0)
            all_levels = torch.cat([all_levels, _levels], dim=0)
        return all_points, all_levels

    def weed_out_by_level(
        self,
        anchors: torch.Tensor,
        levels: torch.Tensor,
        cameras: Union[Cameras, torch.Tensor],
        visibility_threshold: Optional[float] = None,
        max_chunk_size: int = 1 << 24,
    ):
        if visibility_threshold is None:
            visibility_threshold = self.visibility_threshold
        device = cameras.device if isinstance(cameras, torch.Tensor) else cameras.R.device

        anchors, levels = map(lambda x: x.cuda(), (anchors, levels))
        if max_chunk_size > 0:
            chunk_size = self.max_power_of_2(max_chunk_size // len(cameras))
            count = anchors.new_zeros((anchors.shape[0],))
            for st in range(0, anchors.shape[0], chunk_size):
                ed = min(st + chunk_size, anchors.shape[0])
                _anchors, _levels = anchors[st:ed], levels[st:ed]
                pred_levels = self.predict_level(_anchors, cameras)
                int_levels = self.map_to_int_level(pred_levels, self.max_level)
                count[st:ed].copy_((_levels.unsqueeze(0) <= int_levels).sum(dim=0).float())
            count /= len(cameras)
        else:
            pred_levels = self.predict_level(anchors, cameras)
            int_levels = self.map_to_int_level(pred_levels, self.max_level)
            count = (levels.unsqueeze(0) <= int_levels).sum(dim=0).float()
        mask = count > visibility_threshold
        return mask.to(device)

    @staticmethod
    def max_power_of_2(n: int) -> int:
        assert n >= 1
        power, next = 1, 2
        while next <= n:
            power = next
            next *= 2
        return power
