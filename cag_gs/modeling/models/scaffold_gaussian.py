from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import inverse_sigmoid

from ..modules.neural_decoder import NeuralDecoder, NeuralDecoderModule
from ..modules.voxel_grid import VoxelGrid, VoxelGridModule
from ..renders.implicit_renderer import ImplicitRenderingUtils
from ..utils.implicit_wrappers import AnchorFiltering, CameraWrapper, NeuralGaussianWrapper


@dataclass
class ScaffoldOptimization:
    means_lr_init: float = 0.0

    offsets_lr_init: float = 1e-2

    offsets_lr_scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {
                "lr_final": 0.0001,
                "max_steps": None,
            },
        }
    )

    scales_lr: float = 7e-3

    features_lr: float = 7.5e-3

    spatial_lr_scale: float = -1

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})


@dataclass
class ImplicitGaussian(Gaussian):
    n_offsets: int = 10

    feature_dim: int = 32

    neural_decoder: NeuralDecoder = field(default_factory=lambda: NeuralDecoder())


@dataclass
class ScaffoldGaussian(ImplicitGaussian):
    update_depth: int = 3

    update_init_factor: int = 16

    update_hierachy_factor: int = 4

    voxel_grid: VoxelGrid = field(default_factory=lambda: VoxelGrid())

    optimization: ScaffoldOptimization = field(default_factory=lambda: ScaffoldOptimization())

    def instantiate(self, **kwargs):
        return ScaffoldGaussianModel(self, **kwargs)


class ImplicitModelMixin:
    gaussians: nn.ParameterDict

    neural_decoder: NeuralDecoderModule

    _neural_gaussians: "NeuralGaussianWrapper" = NeuralGaussianWrapper()

    def generate_neural_gaussians(self, cameras: Cameras) -> NeuralGaussianWrapper:
        # select visible anchors
        anchor_filter: AnchorFiltering = self.filter_anchors(cameras)

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
        view_features = self.neural_decoder.encode_view(anchors, cam_centers)

        # decode opacities, covariances, colors
        features = self.neural_decoder.decode_feature_bank(view_features, features)
        opacities = self.neural_decoder.decode_opacity(view_features, features).clamp(max=1.0)
        opacities = anchor_filter.apply_prog(opacities).reshape(-1, 1)
        covariances = self.neural_decoder.decode_covariance(view_features, features).reshape(-1, 7)
        colors = self.neural_decoder.decode_color(view_features, features).reshape(-1, 3)

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

    def compute_features(self, valid_camera_ids: torch.Tensor, valid_anchor_ids: torch.Tensor):
        return self.get_features[valid_anchor_ids]

    def get_features_for_rasterization(self, anchor_ids: torch.Tensor):
        return self.gaussians["features"][anchor_ids]

    def pre_activate_all_properties(self):
        pass

    @property
    def max_sh_degree(self):
        return 0

    @property
    def get_anchors(self) -> torch.Tensor:
        return self.gaussians["means"]

    @property
    def get_offsets(self) -> torch.Tensor:
        return self.gaussians["offsets"]

    @property
    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.gaussians["scales"])

    @property
    def get_features(self) -> torch.Tensor:
        return self.gaussians["features"]

    @property
    def get_levels(self) -> torch.Tensor:
        return self.gaussians["levels"]

    @property
    def get_extra_levels(self) -> torch.Tensor:
        return self.gaussians["extra_levels"]

    @property
    def n_anchors(self) -> int:
        return self.get_anchors.shape[0]

    @property
    def n_offsets(self) -> int:
        return self.config.n_offsets

    @property
    def neural_gaussians(self) -> Optional[NeuralGaussianWrapper]:
        _neural_gaussians: NeuralGaussianWrapper = getattr(self, "_neural_gaussians", None)
        if _neural_gaussians is not None and _neural_gaussians.n_gaussians > 0:
            return _neural_gaussians
        return None

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.gaussians["means"]

    @property
    def get_anchor(self) -> torch.Tensor:
        if self.neural_gaussians is not None:
            return self.neural_gaussians.means
        return None

    @property
    def get_scaling(self) -> torch.Tensor:
        if self.neural_gaussians is not None:
            return self.neural_gaussians.scales
        return None

    @property
    def get_rotation(self) -> torch.Tensor:
        if self.neural_gaussians is not None:
            return self.neural_gaussians.quats
        return None

    @property
    def get_opacity(self) -> torch.Tensor:
        if self.neural_gaussians is not None:
            return self.neural_gaussians.opacities
        return None

    @property
    def get_color(self) -> torch.Tensor:
        if self.neural_gaussians is not None:
            return self.neural_gaussians.colors
        return None

    @property
    def is_lod(self) -> bool:
        return "levels" in self.gaussians

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(opacities)


class ScaffoldGaussianModel(GaussianModel, ImplicitModelMixin):
    def __init__(self, config: ScaffoldGaussian):
        super().__init__()
        self.config = config

        self._names = ("means", "offsets", "scales", "features")

    @torch.no_grad()
    def filter_anchors(self, cameras: Cameras, **kwargs) -> AnchorFiltering:
        """
        :returns anchor_mask: [n_cameras, n_anchors], indicating which anchors are visible in each camera
        """
        anchors = self.get_anchors
        scales = self.get_scales[:, :3]

        anchor_mask = kwargs.get("anchor_mask", None)
        if anchor_mask is None:
            anchor_mask = anchors.new_ones((len(cameras), len(anchors)), dtype=torch.bool)

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

        return AnchorFiltering(anchor_mask=out_mask)

    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        from simple_knn._C import distCUDA2

        # setup properties
        points = torch.from_numpy(xyz).float()
        self.voxel_grid = self.config.voxel_grid.instantiate(points=points, **kwargs)

        anchors = self.voxel_grid.voxelize(points)
        offsets = anchors.new_zeros((anchors.shape[0], self.config.n_offsets, 3))
        dist2 = torch.clamp_min(distCUDA2(anchors.cuda()), 0.0000001)
        scales = self.scale_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 6)
        features = anchors.new_zeros((anchors.shape[0], self.config.feature_dim))

        property_dict = {
            "means": nn.Parameter(anchors, requires_grad=True),
            "offsets": nn.Parameter(offsets, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "features": nn.Parameter(features, requires_grad=True),
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
        self.voxel_grid = self.config.voxel_grid.instantiate(points=None, **kwargs)

        anchors = torch.zeros((n, 3), dtype=torch.float32)
        offsets = torch.zeros((n, self.config.n_offsets, 3), dtype=torch.float32)
        scales = torch.zeros((n, 6), dtype=torch.float32)
        features = torch.zeros((n, self.config.feature_dim), dtype=torch.float32)
        property_dict = {
            "means": nn.Parameter(anchors, requires_grad=True),
            "offsets": nn.Parameter(offsets, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "features": nn.Parameter(features, requires_grad=True),
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

        return optimizers, schedulers

    def _add_optimizer_after_backward_hook_if_available(self, optimizer, pl_module):
        hook = getattr(optimizer, "on_after_backward", None)
        if hook is None:
            return
        pl_module.on_after_backward_hooks.append(hook)

    def get_property_names(self):
        return self._names
