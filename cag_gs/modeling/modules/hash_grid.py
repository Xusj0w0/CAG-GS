from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn

from internal.configs.instantiate_config import InstantiatableConfig
from internal.optimizers import Adam
from internal.schedulers import ExponentialDecayScheduler

from .embeddings import (MLPWithHashEncoding, MLPWithMixedHashEncoding,
                         initialize_weights)


@dataclass
class HashGridOptimization:
    lr_init: float = 5e-3
    lr_final: float = 5e-5
    max_steps: Optional[int] = None


@dataclass
class HashGrid(InstantiatableConfig):
    num_levels: int = 5
    min_res: int = 2 << 8
    max_res: int = 2 << 12
    log2_hashmap_size: int = 15
    features_per_level: int = 4

    use_mixed: bool = False
    num_levels_2d: int = 8
    min_res_2d: int = 2 << 8
    max_res_2d: int = 2 << 15
    log2_hashmap_size_2d: int = 15
    features_per_level_2d: int = 4

    tcnn: bool = True
    hash_init_scale: float = 1e-2
    interpolation: Literal["linear", "nearest", "smoothstep"] = "linear"

    mlp_n_layers: int = 2
    mlp_h_dim: int = 64
    mlp_out_dim: int = -1
    activation: Literal["relu", "sigmoid", "tanh", "none"] = "relu"
    out_activation: Literal["relu", "sigmoid", "tanh", "none"] = "none"

    optimization: HashGridOptimization = field(default_factory=lambda: HashGridOptimization())

    def instantiate(self, *args, **kwargs):
        return HashGridModule(self, *args, **kwargs)


class HashGridModule(nn.Module):
    def __init__(self, config: HashGrid, *args, **kwargs):
        super().__init__()
        self.config = config

    def forward(self, points: torch.Tensor):
        return self.model(points)

    def setup(self, stage: str, pl_module=None):
        if stage == "fit":
            params = {
                "num_levels": self.config.num_levels,
                "min_res": self.config.min_res,
                "max_res": self.config.max_res,
                "log2_hashmap_size": self.config.log2_hashmap_size,
                "features_per_level": self.config.features_per_level,
                "hash_init_scale": self.config.hash_init_scale,
                "interpolation": self.config.interpolation,
                "num_layers": self.config.mlp_n_layers,
                "layer_width": self.config.mlp_h_dim,
                "out_dim": self.config.mlp_out_dim,
                "activation": self._activation_str_to_nn_module(self.config.activation),
                "out_activation": self._activation_str_to_nn_module(self.config.out_activation),
                "implementation": "tcnn" if self.config.tcnn else "torch",
            }
            if self.config.use_mixed:
                self.model = MLPWithMixedHashEncoding(
                    num_levels_2d=self.config.num_levels_2d,
                    min_res_2d=self.config.min_res_2d,
                    max_res_2d=self.config.max_res_2d,
                    log2_hashmap_size_2d=self.config.log2_hashmap_size_2d,
                    features_per_level_2d=self.config.features_per_level_2d,
                    **params,
                )
            else:
                self.model = MLPWithHashEncoding(**params)

        initialize_weights(self)

    def training_setup(self, pl_module, **kwargs):
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = pl_module.trainer.max_steps

        optimization = self.config.optimization

        optimizer = Adam().instantiate(
            [{"params": self.model.parameters(), "name": "hash_grid", "lr": optimization.lr_init}],
            lr=optimization.lr_init,
            eps=1e-8,
        )
        scheduler = ExponentialDecayScheduler(lr_final=optimization.lr_final, max_steps=optimization.max_steps)
        scheduler = scheduler.instantiate().get_scheduler(optimizer, lr_init=optimization.lr_init)

        return [optimizer], [scheduler]

    def _activation_str_to_nn_module(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "none":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation}")
