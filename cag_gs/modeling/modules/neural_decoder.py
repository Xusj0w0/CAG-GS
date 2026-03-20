import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from internal.cameras.cameras import Camera, Cameras
from internal.models.gaussian import Gaussian, GaussianModel
from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler

from .embeddings import MLP, SHEncoding, initialize_weights


@dataclass
class NeuralDecoderOptimization:
    opacity_lr_init: float = 2e-3
    opacity_lr_final: float = 2e-5

    covariance_lr_init: float = 4e-3
    covariance_lr_final: float = 4e-5

    color_lr_init: float = 8e-3
    color_lr_final: float = 5e-5

    feature_bank_lr_init: float = 1e-2
    feature_bank_lr_final: float = 1e-5

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})
    scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {"max_steps": None},
        }
    )


@dataclass
class NeuralDecoder:
    color_mode: Literal["rgb", "shs"] = "rgb"  # only support rgb now

    use_dist: bool = False
    """ whether to use distance in view-dependent encoding """

    view_sh_level: int = 0
    """
    SH level for view-dependent encoding:
    - 0: no sh encoding, [B, 3]
    - D: enable sh encoding, [B, D^2]. **D <= 4**
    """
    view_dim: int = field(init=False)

    n_layers: int = 2
    hidden_dim: int = 32
    use_feature_bank: bool = False

    tcnn: bool = False

    optimization: NeuralDecoderOptimization = field(default_factory=lambda: NeuralDecoderOptimization())

    def __post_init__(self):
        if self.view_sh_level == 0:
            self.view_dim = 3
        elif self.view_sh_level <= 4:
            self.view_dim = (self.view_sh_level + 1) ** 2
        else:
            raise ValueError("SH level should be <= 4")
        if self.use_dist:
            self.view_dim += 1

        if self.color_mode == "shs":
            raise NotImplementedError("SH color mode is not implemented yet")

    def instantiate(self, feature_dim: int, n_offsets: int, n_appearance_embeddings: Optional[int] = None, **kwargs):
        return NeuralDecoderModule(
            self,
            feature_dim=feature_dim,
            n_offsets=n_offsets,
            n_appearance_embeddings=n_appearance_embeddings,
            **kwargs,
        )


class NeuralDecoderModule(nn.ModuleDict):
    def __init__(
        self,
        config: NeuralDecoder,
        feature_dim: int,
        n_offsets: int,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.n_offsets = n_offsets

        self._names = ("opacity", "covariance", "color", "feature_bank")

    def encode_view(self, points: torch.Tensor, cam_centers: torch.Tensor, **kwargs):
        viewdir = cam_centers - points
        dist = torch.norm(viewdir, dim=-1, keepdim=True)
        view_encoded = viewdir / (dist + 1e-7)
        if self.config.view_sh_level > 0:
            shape = viewdir.shape
            view_encoded = self.viewdir_encoding(view_encoded.reshape(-1, shape[-1])).reshape(shape)
        if self.config.use_dist:
            view_encoded = torch.cat([view_encoded, dist], dim=-1)
        return view_encoded

    def decode_feature_bank(self, view_features: torch.Tensor, features: torch.Tensor, **kwargs):
        """
        :params features: [N, D]
        :params view_features: [N, C]
        """
        if not hasattr(self, "feature_bank"):
            return features
        bank_weight = F.softmax(self["feature_bank"](view_features), dim=-1)  # [N, 3]
        features = features  # [N, D]
        features = (
            features[..., ::4].repeat(1, 4) * bank_weight[..., 0:1]
            + features[..., ::2].repeat(1, 2) * bank_weight[..., 1:2]
            + features[..., :] * bank_weight[..., 2:3]
        )
        return features

    def decode_opacity(self, view_features: torch.Tensor, features: torch.Tensor, **kwargs):
        """
        :returns opacities: [N, n_offsets]
        """
        opacities = self["opacity"](features)
        return opacities.reshape(-1, self.n_offsets)

    def decode_covariance(self, view_features: torch.Tensor, features: torch.Tensor, **kwargs):
        """
        :returns covariances: [N, n_offsets, 7]
        """
        input_tensor = torch.cat([view_features, features], dim=-1)
        covariance = self["covariance"](input_tensor)
        return covariance.reshape(-1, self.n_offsets, 7)

    def decode_color(self, view_features: torch.Tensor, features: torch.Tensor, **kwargs):
        """
        :returns colors: [N, n_offsets, 3]
        """
        input_tensor = torch.cat([view_features, features], dim=-1)
        color = self["color"](input_tensor)
        return color.reshape(-1, self.n_offsets, 3)

    def setup(self):
        if self.config.view_sh_level > 0:
            self.viewdir_encoding = SHEncoding(
                levels=self.config.view_sh_level,
                implementation="tcnn" if self.config.tcnn else "torch",
            )
        else:
            self.viewdir_encoding = nn.Identity()

        shared_params = {
            "num_layers": self.config.n_layers,
            "layer_width": self.config.hidden_dim,
            "implementation": "tcnn" if self.config.tcnn else "torch",
        }

        # opacity
        opacity_decoder = MLP(
            in_dim=self.feature_dim,  # not conditioned on view
            out_dim=self.n_offsets,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            **shared_params,
        )

        # covariance
        covariance_decoder = MLP(
            in_dim=self.config.view_dim + self.feature_dim,
            out_dim=7 * self.n_offsets,
            activation=nn.ReLU(),
            out_activation=None,
            **shared_params,
        )

        # color
        color_decoder = MLP(
            in_dim=self.config.view_dim + self.feature_dim,
            out_dim=3 * self.n_offsets,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            **shared_params,
        )

        # feature bank
        feature_bank_decoder = MLP(
            in_dim=self.config.view_dim,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            **shared_params,
        )

        self["opacity"] = opacity_decoder
        self["covariance"] = covariance_decoder
        self["color"] = color_decoder
        if self.config.use_feature_bank:
            self["feature_bank"] = feature_bank_decoder

        initialize_weights(self)

    def training_setup(self, pl_module: lightning.LightningModule, **kwargs):
        if self.config.optimization.scheduler.max_steps is None:
            self.config.optimization.scheduler.max_steps = pl_module.trainer.max_steps

        optimization = self.config.optimization
        optimizer_factory = optimization.optimizer
        scheduler_factory = optimization.scheduler

        optimizers, schedulers = [], []

        def configure(params, lr_init: float, lr_final: float):
            if lr_init <= 0.0:
                return
            optimizer = optimizer_factory.instantiate(params, lr=lr_init, eps=1e-8)
            self._add_optimizer_after_backward_hook_if_available(optimizer, pl_module)

            scheduler = deepcopy(scheduler_factory)
            scheduler.lr_final = lr_final
            scheduler = scheduler.instantiate().get_scheduler(optimizer, lr_init=lr_init)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        # opacity
        configure(
            [
                {
                    "params": self["opacity"].parameters(),
                    "lr": optimization.opacity_lr_init,
                    "name": "opacity_decoder",
                }
            ],
            lr_init=optimization.opacity_lr_init,
            lr_final=optimization.opacity_lr_final,
        )
        # covariance
        configure(
            [
                {
                    "params": self["covariance"].parameters(),
                    "lr": optimization.covariance_lr_init,
                    "name": "covariance_decoder",
                }
            ],
            lr_init=optimization.covariance_lr_init,
            lr_final=optimization.covariance_lr_final,
        )
        # color
        configure(
            [
                {
                    "params": self["color"].parameters(),
                    "lr": optimization.color_lr_init,
                    "name": "color_decoder",
                }
            ],
            lr_init=optimization.color_lr_init,
            lr_final=optimization.color_lr_final,
        )
        # feature bank
        if hasattr(self, "feature_bank"):
            configure(
                [
                    {
                        "params": self["feature_bank"].parameters(),
                        "lr": optimization.feature_bank_lr_init,
                        "name": "feature_bank",
                    }
                ],
                lr_init=optimization.feature_bank_lr_init,
                lr_final=optimization.feature_bank_lr_final,
            )
        return optimizers, schedulers

    def train(self, mode=True):
        for name in self._names:
            if name in self:
                self[name].train(mode)
        return super().train(mode)

    def eval(self):
        for name in self._names:
            if name in self:
                self[name].eval()
        return super().eval()

    def _add_optimizer_after_backward_hook_if_available(self, optimizer, pl_module):
        hook = getattr(optimizer, "on_after_backward", None)
        if hook is None:
            return
        pl_module.on_after_backward_hooks.append(hook)
