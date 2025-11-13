from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class WeightSchedulerBase:
    pass


@dataclass
class WeightScheduler(WeightSchedulerBase):
    init: float = 1.0

    final: float = 1.0

    max_steps: Optional[int] = None

    mode: Literal["constant", "log", "exp", "linear"] = "constant"

    start_iter: int = 0

    end_iter: Optional[int] = None

    def check(self, max_steps: int):
        if self.max_steps is None:
            self.max_steps = max_steps
        if self.end_iter is None:
            self.end_iter = self.max_steps

    def __call__(self, step: int) -> float:
        if step < self.start_iter or step >= self.end_iter:
            return 0.0
        if self.init <= 0.0 and self.final <= 0.0:
            return 0.0
        return self._calculate_weight(step)

    def _calculate_weight(self, step: int) -> float:
        t = np.clip(step / self.max_steps, 0.0, 1.0)
        if self.mode == "constant":
            return self.init
        elif self.mode == "linear":
            return self.init * (1.0 - t) + self.final * t
        elif self.mode == "exp":
            return self.init * ((self.final / self.init) ** t)
        elif self.mode == "log":
            return np.exp(np.log(self.init) * (1.0 - t) + np.log(self.final) * t)
        else:
            raise ValueError(f"unsupported mode")


@dataclass
class DepthLoss:
    type: Literal["l1", "l2", "kl"] = "l1"
    """Type of depth loss function."""

    ssim_weight: float = 0.2

    normalized: bool = False

    median_normalized: bool = False

    mean_normalized: bool = False

    def __call__(self, depth: torch.Tensor, gt_depth: torch.Tensor):
        if self.normalized:
            with torch.no_grad():
                max_depth = depth.max(dim=[-1, -2], keepdim=True).values
                min_depth = depth.min(dim=[-1, -2], keepdim=True).values
            depth = (depth - min_depth) / (max_depth - min_depth + 1e-6)
        elif self.median_normalized:
            median = torch.median(gt_depth, dim=[-1, -2], keepdim=True).values
            depth = depth / median
            gt_depth = gt_depth / median
        elif self.mean_normalized:
            mean = torch.mean(gt_depth, dim=[-1, -2], keepdim=True)
            depth = depth / mean
            gt_depth = gt_depth / mean
        return self._loss(depth, gt_depth)

    def _loss(self, depth: torch.Tensor, gt_depth: torch.Tensor):
        if self.type == "l1":
            return self._depth_l1_loss(depth, gt_depth)
        # elif self.type == "l1+ssim":
        #     return self._depth_l1_and_ssim_loss(depth, gt_depth)
        elif self.type == "l2":
            return self._depth_l2_loss(depth, gt_depth)
        elif self.type == "kl":
            return self._depth_kl_loss(depth, gt_depth)
        else:
            raise ValueError(f"Unknown depth loss type: {self.type}")

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    # def _depth_l1_and_ssim_loss(self, a, b):
    #     l1_loss = self._depth_l1_loss(a, b)
    #     ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])
    #     return (1 - self.ssim_weight) * l1_loss + self.ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass
