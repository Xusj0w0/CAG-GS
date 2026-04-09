import json
import math
import os
import os.path as osp
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from gsplat.rasterize_to_weights import rasterize_to_weights
from matplotlib.pyplot import cm
from torch_scatter import scatter_sum
from tqdm.auto import tqdm

from internal.cameras import Cameras
from internal.configs.instantiate_config import InstantiatableConfig
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataparsers.dataparser import (DataParserOutputs, ImageSet,
                                             PointCloud)
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import build_rotation

from .implicit_wrappers import NeuralGaussianWrapper, ProjectionWrapper


def torch2numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@dataclass
class MinMaxBoundingBox:
    min: torch.Tensor  # [2]
    max: torch.Tensor  # [2]


@dataclass
class MinMaxBoundingBoxes:
    min: torch.Tensor  # [N, 2]
    max: torch.Tensor  # [N, 2]

    def __getitem__(self, item):
        return MinMaxBoundingBox(
            min=self.min[item],
            max=self.max[item],
        )

    def to(self, *args, **kwargs):
        self.min = self.min.to(*args, **kwargs)
        self.max = self.max.to(*args, **kwargs)
        return self


@dataclass
class PartitionCoordinates:
    id: torch.Tensor  # [N_partitions, 2]
    xy: torch.Tensor  # [N_partitions, 2]
    size: torch.Tensor  # [N_partitions, 2]

    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, item):
        return self.id[item], self.xy[item], self.size[item]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.id[idx], self.xy[idx], self.size[idx]

    def get_bounding_boxes(self, enlarge: Union[float, torch.Tensor] = 0.0) -> MinMaxBoundingBoxes:
        xy_min = self.xy - (enlarge * self.size)  # [N_partitions, 2]
        xy_max = self.xy + self.size + (enlarge * self.size)  # [N_partitions, 2]
        return MinMaxBoundingBoxes(
            min=xy_min,
            max=xy_max,
        )

    def extend_to_bbox(self, bbox: MinMaxBoundingBox):
        _xymin = self.xy.clone()
        _xymax = _xymin + self.size.clone()
        _id = self.id.clone()

        x_dim, y_dim = self.id[:, 0].max().item() + 1, self.id[:, 1].max().item() + 1
        for block_idx in range(len(self)):
            if _id[block_idx][0] == 0:
                _xymin[block_idx][0] = bbox.min[0]
            if _id[block_idx][0] == x_dim - 1:
                _xymax[block_idx][0] = bbox.max[0]
            if _id[block_idx][1] == 0:
                _xymin[block_idx][1] = bbox.min[1]
            if _id[block_idx][1] == y_dim - 1:
                _xymax[block_idx][1] = bbox.max[1]

            return PartitionCoordinates(id=_id, xy=_xymin, size=_xymax - _xymin)


@dataclass
class SceneConfig(InstantiatableConfig):
    dataset_path: str = ""

    coarse_model_path: str = ""

    transforms: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 6)
    "Scene transformation, in [qw, qx, qy, qz, tx, ty, tz] format"

    partition_dim: List[int] = field(default_factory=lambda: [1, 1])

    scene_bbox_enlarge_by_pts: float = 0.0

    scene_bbox_outlier_by_pts: float = 0.005

    scene_bbox_enlarge_by_campos: float = 0.2

    bbox_enlarge_by_campos: float = 0.2
    "Enlarge block bbox for camera position based assignment"

    bbox_enlarge_by_camvis: float = 0.2
    "Enlarge block bbox for camera visibility computation"

    camera_visibility_threshold: float = 0.25

    def __post_init__(self):
        assert osp.exists(self.dataset_path) and osp.exists(
            self.coarse_model_path
        ), "Dataset path and checkpoint path must exist"
        assert len(self.partition_dim) == 2, "Partition dimension must be a list of two integers"
        assert len(self.transforms) == 7, "Transforms must be a list of seven floats [qw, qx, qy, qz, tx, ty, tz]"

    def instantiate(self, *args, **kwargs):
        return PartitionableScene(config=self, *args, **kwargs)


class PartitionableScene:
    def __init__(self, config: SceneConfig, **kwargs):
        self.config = config
        self.device = torch.device("cuda")

    def run(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        # Save config
        with open(osp.join(output_path, "config.yaml"), "w") as f:
            yaml.safe_dump(asdict(self.config), f, indent=4, sort_keys=False)

        # Load scene
        ckpt_path = GaussianModelLoader.search_load_file(self.config.coarse_model_path)
        gaussian_model, renderer, ckpt, dataparser_outputs = self.load_scene()
        image_set = dataparser_outputs.train_set
        pcd = dataparser_outputs.point_cloud

        # Apply transformation to camera positions
        campos = image_set.cameras.camera_center.to(self.device)
        campos_transformed = campos @ self.rotation.T + self.translation

        # Compute scene bounding box and division
        scene_bbox = self.get_bounding_box_by_campos(campos_transformed)
        campos_bbox = MinMaxBoundingBox(
            min=torch.min(campos_transformed[..., :2], dim=0).values,
            max=torch.max(campos_transformed[..., :2], dim=0).values,
        )
        partition_coords = self.balanced_camera_based_division(campos_transformed, campos_bbox)

        fig, ax = plt.subplots()
        self.set_plot_ax_limit(ax, scene_bbox)
        os.makedirs(osp.join(output_path, "figures"), exist_ok=True)
        # Plot scene and division
        _, scene_bbox_obj = self.plot_scene(ax, pcd, scene_bbox)
        fig.savefig(osp.join(output_path, "figures", "scene.png"), dpi=300)
        scene_bbox_obj.remove()
        block_bbox_objs = self.plot_scene_division(ax, partition_coords)
        fig.savefig(osp.join(output_path, "figures", "scene_division.png"), dpi=300)
        for obj in block_bbox_objs:
            obj.remove()

        # Camera position based assignment
        campos_assign = self.is_in_bboxes(
            partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_campos), campos_transformed
        )
        # Camera visibility based assignment
        cam_vis = self.compute_camera_visibility(partition_coords, image_set.cameras, gaussian_model, renderer)
        camvis_assign = cam_vis > self.config.camera_visibility_threshold
        camera_assign = torch.logical_or(campos_assign, camvis_assign)
        print("Num cameras assigned to each block: {}".format(camera_assign.sum(dim=1).tolist()))

        # Plot camera assignment
        for block_idx in range(len(partition_coords)):
            block_objs = self.plot_block(ax, block_idx, partition_coords, campos_transformed, camera_assign)
            fig.savefig(osp.join(output_path, "figures", "block_{:03d}.png".format(block_idx)), dpi=300)
            for obj in block_objs:
                obj.remove()

        plt.close(fig)

        # Save partitions
        self.save_partitions(
            output_path, ckpt_path, scene_bbox, partition_coords, gaussian_model, image_set, camera_assign
        )
        partition_info = {
            "scene_bbox": asdict(scene_bbox),
            "partition_coords": asdict(partition_coords),
            "campos_assign": campos_assign,
            "camvis_assign": camvis_assign,
            "camera_visibility": cam_vis,
        }
        torch.save(partition_info, osp.join(output_path, "partition_info.pt"))

    def balanced_camera_based_division(self, campos: torch.Tensor, scene_bbox: MinMaxBoundingBox):
        num_cams = len(campos)
        x_dim, y_dim = self.config.partition_dim
        # Example 3x4 partition:
        # 3 7 11      (0,3) (1,3) (2,3)
        # 2 6 10      (0,2) (1,2) (2,2)
        # 1 5 9       (0,1) (1,1) (2,1)
        # 0 4 8       (0,0) (1,0) (2,0)

        # Divide cameras along x-axis
        num_cameras_per_column = math.ceil(num_cams / x_dim)
        _, x_sort_indices = torch.sort(campos[:, 0], dim=0)
        x_splits = [0.0 for _ in range(x_dim - 1)]
        y_splits = [[0.0 for _ in range(y_dim - 1)] for _ in range(x_dim)]
        for i, x_st in enumerate(range(0, num_cams, num_cameras_per_column)):
            x_ed = min(x_st + num_cameras_per_column, num_cams)
            cam_ids_in_col = x_sort_indices[x_st:x_ed]
            campos_in_col = campos[cam_ids_in_col]

            # Determine x split
            if i != 0:
                x_splits[i - 1] = 0.5 * (campos_in_col[:, 0].min() + prev_col_max_x)
            prev_col_max_x = campos_in_col[:, 0].max()

            # Divide cameras along y-axis
            _, y_sort_indices = torch.sort(campos_in_col[:, 1], dim=0)
            num_cams_in_col = len(campos_in_col)
            num_cams_per_block = math.ceil(num_cams_in_col / y_dim)
            for j, y_st in enumerate(range(0, num_cams_in_col, num_cams_per_block)):
                y_ed = min(y_st + num_cams_per_block, num_cams_in_col)
                cam_ids_in_block = cam_ids_in_col[y_sort_indices[y_st:y_ed]]
                campos_in_block = campos[cam_ids_in_block]

                if j != 0:
                    y_splits[i][j - 1] = 0.5 * (campos_in_block[:, 1].min() + prev_block_max_y)
                prev_block_max_y = campos_in_block[:, 1].max()

        # Build partition coords
        id_tensor, xy_tensor, size_tensor = (
            torch.empty((0, 2), device=self.device, dtype=torch.long),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
        )
        for i in range(x_dim):
            for j in range(y_dim):
                if i == 0:
                    x_min, x_max = scene_bbox.min[0], x_splits[0]
                elif i == x_dim - 1:
                    x_min, x_max = x_splits[i - 1], scene_bbox.max[0]
                else:
                    x_min, x_max = x_splits[i - 1], x_splits[i]
                if j == 0:
                    y_min, y_max = scene_bbox.min[1], y_splits[i][0]
                elif j == y_dim - 1:
                    y_min, y_max = y_splits[i][j - 1], scene_bbox.max[1]
                else:
                    y_min, y_max = y_splits[i][j - 1], y_splits[i][j]

                id_tensor = torch.cat([id_tensor, torch.tensor([[i, j]]).to(id_tensor)], dim=0)
                xy_tensor = torch.cat([xy_tensor, torch.tensor([[x_min, y_min]]).to(xy_tensor)], dim=0)
                size_tensor = torch.cat(
                    [
                        size_tensor,
                        torch.tensor([[x_max - x_min, y_max - y_min]]).to(size_tensor),
                    ],
                    dim=0,
                )

        return PartitionCoordinates(id=id_tensor, xy=xy_tensor, size=size_tensor)

    @torch.no_grad()
    def compute_camera_visibility(
        self, partition_coords: PartitionCoordinates, cameras: Cameras, gaussian_model, renderer
    ):
        cam_vis = torch.zeros((len(partition_coords), len(cameras)), device=self.device, dtype=torch.float32)

        n_anchors, n_offsets = gaussian_model.get_anchors.shape[0], gaussian_model.n_offsets
        bboxes = partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_camvis)
        gs_means = gaussian_model.get_anchors.detach().clone().to(self.device)
        anchors = gs_means @ self.rotation.T + self.translation
        is_in_bboxes = torch.zeros((len(partition_coords), n_anchors), dtype=torch.bool, device=self.device)
        for i in range(len(partition_coords)):
            bbox = bboxes[i]
            is_in_bboxes[i] = torch.logical_and(
                (anchors[..., :2] > bbox.min[..., :2]).all(dim=-1), (anchors[..., :2] < bbox.max[..., :2]).all(dim=-1)
            )

        bg_color = torch.zeros((3,), device=self.device, dtype=torch.float32)
        for cam_idx, camera in tqdm(enumerate(cameras), total=len(cameras), desc="Computing camera visibility"):
            camera = camera.to_device(self.device)
            outputs = renderer(camera, gaussian_model, bg_color, render_types=[])
            gaussians: NeuralGaussianWrapper = outputs["neural_gaussians"]
            projections: ProjectionWrapper = outputs["projections"]

            image_width, image_height = int(camera.width), int(camera.height)
            tile_size = getattr(renderer.config, "block_size", 16)
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
            anchor_weights = scatter_sum(blend_weights, projections.anchor_ids, dim_size=n_anchors)
            total_weights = anchor_weights.sum()

            for part_idx in range(len(partition_coords)):
                cam_vis[part_idx, cam_idx] = anchor_weights[is_in_bboxes[part_idx]].sum() / total_weights
        return cam_vis

    def save_partitions(
        self,
        output_path: str,
        ckpt_path: str,
        scene_bbox: MinMaxBoundingBox,
        partition_coords: PartitionCoordinates,
        gaussian_model,
        image_set: ImageSet,
        camera_assign: torch.Tensor,
    ):
        metadata = {}
        metadata["checkpoint_path"] = ckpt_path
        metadata["scene"] = {
            "transforms": self.config.transforms,
            "bbox": scene_bbox.min.tolist() + scene_bbox.max.tolist(),
        }
        metadata["blocks"] = []

        for block_idx, (block_id, block_xy, block_size) in enumerate(tqdm(partition_coords, desc="Saving partitions")):
            block_dir = osp.join(output_path, "partitions", "block_{:03d}".format(block_idx))
            os.makedirs(block_dir, exist_ok=True)
            valid_cam_ids = camera_assign[block_idx].nonzero().squeeze().tolist()

            # Save metadata
            metadata["blocks"].append(
                {
                    "id": block_idx,
                    "name": f"block_{block_idx:03d}",
                    "block_id": block_id.tolist(),
                    "bbox": block_xy.tolist() + (block_xy + block_size).tolist(),
                    "n_cameras": len(valid_cam_ids),
                }
            )

            # Save camera to json
            camera_list = []
            for cam_id in valid_cam_ids:
                camera = image_set.cameras[cam_id]
                c2w = torch.linalg.inv(camera.world_to_camera.T)
                camera_list.append(
                    {
                        "id": cam_id,
                        "img_name": image_set.image_names[cam_id],
                        "width": int(camera.width),
                        "height": int(camera.height),
                        "position": c2w[:3, -1].numpy().tolist(),
                        "rotation": c2w[:3, :3].numpy().tolist(),
                        "fx": float(camera.fx),
                        "fy": float(camera.fy),
                        "cx": camera.cx.item(),
                        "cy": camera.cy.item(),
                        "time": camera.time.item() if camera.time is not None else None,
                        "appearance_id": (camera.appearance_id.item() if camera.appearance_id is not None else None),
                        "normalized_appearance_id": (
                            camera.normalized_appearance_id.item()
                            if camera.normalized_appearance_id is not None
                            else None
                        ),
                    }
                )
            with open(osp.join(block_dir, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, separators=(", ", ": "))

            # Write image list
            with open(osp.join(block_dir, "image_list.txt"), "w") as f:
                for cam_id in valid_cam_ids:
                    f.write(f"{image_set.image_names[cam_id]}\n")

        with open(osp.join(output_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, separators=(", ", ": "))

    def load_scene(self):
        ckpt_path = GaussianModelLoader.search_load_file(self.config.coarse_model_path)
        gaussian_model, renderer, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ckpt_path, device=self.device, eval_mode=False, pre_activate=False
        )
        dataparser: Colmap = ckpt["datamodule_hyper_parameters"]["parser"]
        dataparser.split_mode = "reconstruction"
        dataparser.points_from = "sfm"
        dataparser_outputs = dataparser.instantiate(
            path=self.config.dataset_path, output_path=os.getcwd(), global_rank=0
        ).get_outputs()
        return gaussian_model, renderer, ckpt, dataparser_outputs

    def get_bounding_box_by_campos(self, campos: torch.Tensor):
        xy_min = torch.min(campos[..., :2], dim=0).values
        xy_max = torch.max(campos[..., :2], dim=0).values
        if self.config.scene_bbox_enlarge_by_campos > 0.0:
            size = xy_max - xy_min
            enlarge_size = size * self.config.scene_bbox_enlarge_by_campos
            xy_min = xy_min - enlarge_size
            xy_max = xy_max + enlarge_size

        return MinMaxBoundingBox(min=xy_min, max=xy_max)

    def plot_scene(self, ax: plt.Axes, point_cloud: PointCloud, scene_bbox: MinMaxBoundingBox):
        # Sparsify points
        STEP = 32

        # Apply transformation to point cloud
        pts_xyz = torch.from_numpy(point_cloud.xyz[::STEP]).to(self.translation)
        pts_xyz = pts_xyz @ self.rotation.T + self.translation
        pts_rgb = torch.from_numpy(point_cloud.rgb[::STEP]).to(self.translation) / 255.0

        # Plot scene
        _pts_xyz = torch2numpy(pts_xyz)
        _pts_rgb = torch2numpy(pts_rgb)
        pcd_obj = ax.scatter(_pts_xyz[:, 0], _pts_xyz[:, 1], s=0.1, c=_pts_rgb, marker=".")
        # Plot scene bbox
        scene_bbox_min, scene_bbox_max = torch2numpy(scene_bbox.min), torch2numpy(scene_bbox.max)
        scene_bbox_obj = ax.add_artist(
            mpatches.Rectangle(
                (scene_bbox_min[0], scene_bbox_min[1]),
                scene_bbox_max[0] - scene_bbox_min[0],
                scene_bbox_max[1] - scene_bbox_min[1],
                fill=False,
                edgecolor="green",
                linewidth=2.0,
                linestyle="--",
            )
        )
        return [pcd_obj, scene_bbox_obj]

    def plot_scene_division(self, ax: plt.Axes, partition_coords: PartitionCoordinates):
        # plot division
        block_bbox_objs = []
        for block_idx, (part_id, part_xy, part_size) in enumerate(partition_coords):
            block_bbox_min, block_bbox_max = torch2numpy(part_xy), torch2numpy(part_xy + part_size)
            block_bbox_obj = ax.add_artist(
                mpatches.Rectangle(
                    (block_bbox_min[0], block_bbox_min[1]),
                    block_bbox_max[0] - block_bbox_min[0],
                    block_bbox_max[1] - block_bbox_min[1],
                    fill=False,
                    edgecolor=self.COLORLIST[block_idx % self.N_COLORS],
                    linewidth=1.0,
                    linestyle="-",
                )
            )
            block_bbox_objs.append(block_bbox_obj)
        return block_bbox_objs

    def plot_block(
        self,
        ax: plt.Axes,
        block_idx: int,
        partition_coords: PartitionCoordinates,
        campos: torch.Tensor,
        camera_assign: torch.Tensor,
    ):
        color = self.COLORLIST[block_idx % self.N_COLORS]

        # plot block bbox
        part_id, part_xy, part_size = partition_coords[block_idx]
        block_bbox_min, block_bbox_max = torch2numpy(part_xy), torch2numpy(part_xy + part_size)
        block_bbox_obj = ax.add_artist(
            mpatches.Rectangle(
                (block_bbox_min[0], block_bbox_min[1]),
                block_bbox_max[0] - block_bbox_min[0],
                block_bbox_max[1] - block_bbox_min[1],
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                linestyle="-",
            )
        )
        # Plot cameras
        _campos = torch2numpy(campos[camera_assign[block_idx]])
        campos_obj = ax.scatter(_campos[:, 0], _campos[:, 1], s=0.8, c="red", marker="o")
        # Annotate
        annotation_obj = ax.annotate(
            "block #{}: ({}, {}), {} cameras".format(block_idx, part_id[0].item(), part_id[1].item(), _campos.shape[0]),
            xy=(
                block_bbox_min[0] + 0.125 * (block_bbox_max[0] - block_bbox_min[0]),
                block_bbox_min[1] + 0.25 * (block_bbox_max[1] - block_bbox_min[1]),
            ),
            fontsize=8,
        )

        return [block_bbox_obj, campos_obj, annotation_obj]

    def set_plot_ax_limit(self, ax, scene_bbox: MinMaxBoundingBox, enlarge: float = 0.08):
        x_enlarge = (scene_bbox.max[0] - scene_bbox.min[0]) * enlarge
        y_enlarge = (scene_bbox.max[1] - scene_bbox.min[1]) * enlarge
        enlarge = max(x_enlarge, y_enlarge)

        ax.set_xlim(
            [
                (scene_bbox.min[0] - enlarge).item(),
                (scene_bbox.max[0] + enlarge).item(),
            ]
        )
        ax.set_ylim(
            [
                (scene_bbox.min[1] - enlarge).item(),
                (scene_bbox.max[1] + enlarge).item(),
            ]
        )

        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def is_in_bboxes(bboxes: MinMaxBoundingBoxes, points: torch.Tensor) -> torch.Tensor:
        xy_min, xy_max = bboxes.min.unsqueeze(1), bboxes.max.unsqueeze(1)  # [N_partitions, 1, 2]
        points = points[..., :2].unsqueeze(0)  # [1, N, 2]
        is_in_partition = torch.logical_and(
            (points >= xy_min.to(points)).all(dim=-1),
            (points <= xy_max.to(points)).all(dim=-1),
        )  # [N_partitions, N]
        return is_in_partition

    @property
    def rotation(self) -> torch.Tensor:
        if not hasattr(self, "_rotation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            rotation = transforms[:4]
            self._rotation = build_rotation(rotation.unsqueeze(0)).squeeze(0)
        return self._rotation

    @property
    def translation(self) -> torch.Tensor:
        if not hasattr(self, "_translation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            self._translation = transforms[4:]
        return self._translation

    @property
    def COLORLIST(self):
        if not hasattr(self, "_colorlist"):
            self._colorlist = list(iter(cm.rainbow(np.linspace(0, 1, self.N_COLORS))))
        return self._colorlist

    @property
    def N_COLORS(self):
        return 7
