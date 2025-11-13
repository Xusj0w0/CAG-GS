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


@dataclass
class DecoupledAppModelOptimization:
    embedding_lr_init: float = 2e-3
    embedding_lr_final: float = 2e-4

    network_lr_init: float = 1e-3
    network_lr_final: float = 1e-4

    max_steps: Optional[int] = None

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})
    scheduler: Scheduler = field(
        default_factory=lambda: {
            "class_path": "ExponentialDecayScheduler",
            "init_args": {"max_steps": None},
        }
    )


@dataclass
class DecoupledAppearanceModel:
    out_dim: int = 3

    n_appearances: int = -1

    embedding_dim: int = -1
    """ if < 0, disable appearance model """

    optimization: DecoupledAppModelOptimization = field(default_factory=lambda: DecoupledAppModelOptimization())

    def instantiate(self, **kwargs):
        return DecoupledAppearanceModule(self, **kwargs)


class DecoupledAppearanceModule(nn.Module):
    def __init__(self, config: DecoupledAppearanceModel, **kwargs):
        super().__init__()
        self.config = config

    def setup(self, stage: str, pl_module=None, **kwargs):
        if stage == "fit":
            if pl_module is not None:
                if self.config.n_appearances <= 0:
                    max_input_id = 0
                    appearance_group_ids = pl_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                    if appearance_group_ids is not None:
                        for i in appearance_group_ids.values():
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    n_appearances = max_input_id + 1
                    self.config.n_appearances = n_appearances
            if self.config.n_appearances > 0:
                self.embedding = nn.Embedding(self.config.n_appearances, self.config.embedding_dim)
                self.network = AppearanceNetwork(
                    num_input_channels=self.config.embedding_dim + 3,
                    num_output_channels=self.config.out_dim,
                )

    def training_setup(self, pl_module: lightning.LightningModule, **kwargs):
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = pl_module.trainer.max_steps

        optimization = self.config.optimization
        optimizer_factory = optimization.optimizer
        scheduler_factory = optimization.scheduler

        optimizers, schedulers = [], []

        def configure(params, lr_init: float, lr_final: float):
            if lr_init <= 0.0:
                return
            optimizer = optimizer_factory.instantiate(params, lr=lr_init, eps=1e-8)

            scheduler = deepcopy(scheduler_factory)
            scheduler.lr_final = lr_final
            scheduler = scheduler.instantiate().get_scheduler(optimizer, lr_init=lr_init)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

        configure(
            [
                {
                    "params": self.embedding.parameters(),
                    "name": "decoupled_appearance_embedding",
                    "lr": optimization.embedding_lr_init,
                }
            ],
            lr_init=optimization.embedding_lr_init,
            lr_final=optimization.embedding_lr_final,
        )
        configure(
            [
                {
                    "params": self.network.parameters(),
                    "name": "decoupled_appearance_network",
                    "lr": optimization.network_lr_init,
                }
            ],
            lr_init=optimization.network_lr_init,
            lr_final=optimization.network_lr_final,
        )
        return optimizers, schedulers

    def forward(self, render: torch.Tensor, appearance_ids: torch.Tensor, **kwargs):
        """
        :params render: Tensor in BCHW format
        :params appearance_ids: (B,)
        :returns output: enhanced rendered results in BCHW format
        """
        appearance_embeddings = self.embedding(appearance_ids)
        h, w = render.shape[-2:]
        crop_image_down = F.interpolate(render, size=(h // 32, w // 32), mode="bilinear", align_corners=True)
        net_input = torch.cat(
            [crop_image_down, appearance_embeddings[..., None, None].repeat(1, 1, h // 32, w // 32)], dim=1
        )
        output = self.network(net_input, h, w) * render
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.act = nn.Tanh()  # nn.Sigmoid()

    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.act(x)
        return x * 0.5 + 1.0
