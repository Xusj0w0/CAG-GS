import math
import os
import os.path as osp
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal.optimizers import Adam, OptimizerConfig
from internal.schedulers import ExponentialDecayScheduler, Scheduler


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)


@dataclass
class FeatureAdapterOptimization:
    lr_init: float = 5e-3
    lr_final: float = 5e-5
    max_steps: Optional[int] = None


@dataclass
class FeatureAdapter:
    in_dim: int = -1
    out_dim: int = -1
    optimization: FeatureAdapterOptimization = field(default_factory=lambda: FeatureAdapterOptimization())

    def instantiate(self):
        return FeatureAdapterModule(self)


class FeatureAdapterModule(nn.Module):
    def __init__(self, config: FeatureAdapter):
        super().__init__()
        self.config = config

    def setup(self, stage: str, pl_module=None, **kwargs):
        self.network = nn.Sequential(
            nn.Linear(self.config.in_dim, self.config.out_dim, bias=False),
        )
        self.apply(_init_weights)

    def training_setup(self, pl_module: lightning.LightningModule, **kwargs):
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = pl_module.trainer.max_steps

        optimization = self.config.optimization

        optimizers, schedulers = [], []

        def configure(params, lr_init: float, lr_final: float):
            if lr_init <= 0.0:
                return
            optimizer = Adam().instantiate(params, lr=lr_init, eps=1e-8)

            scheduler = ExponentialDecayScheduler(lr_final=lr_final, max_steps=optimization.max_steps)
            scheduler = scheduler.instantiate().get_scheduler(optimizer, lr_init=lr_init)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        configure(
            [
                {
                    "params": self.network.parameters(),
                    "name": "feature_adapter",
                    "lr": optimization.lr_init,
                }
            ],
            lr_init=optimization.lr_init,
            lr_final=optimization.lr_final,
        )
        return optimizers, schedulers

    def forward(self, x: torch.Tensor):
        return self.network(x)
