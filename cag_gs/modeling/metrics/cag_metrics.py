from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import lightning
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.utils import depth_to_normal

from internal.cameras import Camera, Cameras

from ..utils.implicit_wrappers import NeuralGaussianWrapper
from ..utils.loss_utils import WeightScheduler
from .implicit_metrics import ImplicitMetrics, ImplicitMetricsImpl


@dataclass
class CAGMetrics(ImplicitMetrics):
    lambda_flatten: float = 100.0

    dn_from_iter: int = 7_000

    lambda_dn: float = 0.015

    lambda_feat: WeightScheduler = field(default_factory=lambda: WeightScheduler(init=1.0, final=1.0))

    use_grad_weight: bool = True

    def instantiate(self, *args, **kwargs):
        return CAGMetricsImpl(self)


class CAGMetricsImpl(ImplicitMetricsImpl):
    config: CAGMetrics

    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        global_step = pl_module.trainer.global_step + 1
        batch, outputs = self.batchify(batch, outputs)
        cameras, (image_names, gt_images, masks), extra_data = batch
        metrics, pbar = super()._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

        if self.config.lambda_flatten > 0.0:
            gaussians: NeuralGaussianWrapper = outputs["neural_gaussians"]
            scales = gaussians.scales
            if scales.shape[0] > 0:
                flatten_loss = torch.min(scales, dim=-1).values.mean()
            else:
                flatten_loss = torch.tensor(0.0).to(metrics["loss"])

            metrics["loss"] += self.config.lambda_flatten * flatten_loss
            metrics["loss_flatten"] = flatten_loss
            pbar["loss_flatten"] = False

        if self.config.lambda_dn > 0.0 and global_step > self.config.dn_from_iter:
            depth = outputs.get("unbiased_depth", None)
            normal = outputs.get("normal", None)
            if depth is not None and normal is not None:
                cam2worlds = torch.linalg.inv(cameras.world_to_camera.transpose(-1, -2))
                Ks = cameras[0].get_K()[:3, :3].unsqueeze(0).expand(depth.shape[0], -1, -1)
                normal_from_depth = depth_to_normal(
                    depth.squeeze(1)[..., None], cam2worlds, Ks
                ).permute(0, 3, 1, 2)
                normal_error = (normal - normal_from_depth).abs().sum(dim=1)
                if self.config.use_grad_weight:
                    weight = self._get_grad_weight(gt_images)
                    dn_loss = (normal_error * weight).mean()
                else:
                    dn_loss = normal_error.mean()
            else:
                dn_loss = torch.tensor(0.0).to(metrics["loss"])

            metrics["loss"] += self.config.lambda_dn * dn_loss
            metrics["loss_dn"] = dn_loss
            pbar["loss_dn"] = False

        return metrics, pbar

    def _get_grad_weight(self, image: torch.Tensor):
        rgb_grad = image.new_ones((image.shape[0], *image.shape[-2:]))
        grad_x = (image[..., 1:-1, 2:] - image[..., 1:-1, :-2]).abs().mean(dim=1, keepdim=True)
        grad_y = (image[..., 2:, 1:-1] - image[..., :-2, 1:-1]).abs().mean(dim=1, keepdim=True)
        grad = torch.cat([grad_x, grad_y], dim=1).max(dim=1).values
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        rgb_grad[..., 1:-1, 1:-1] = grad
        weight = (1.0 - rgb_grad) ** 2
        return weight
