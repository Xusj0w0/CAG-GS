from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import lightning
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from internal.cameras.cameras import BatchedCameras, Camera
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl

from ...data.dataset_utils import FeatureData, InverseDepthData
from ..models.octree_gaussian import OctreeGaussianModel
from ..models.scaffold_gaussian import ScaffoldGaussianModel
from ..utils.implicit_wrappers import NeuralGaussianWrapper
from ..utils.loss_utils import DepthLoss, WeightScheduler, WeightSchedulerBase


@dataclass
class ImplicitMetrics(VanillaMetrics):
    lambda_dreg: float = 0.01

    lambda_depth: WeightScheduler = field(
        default_factory=lambda: WeightScheduler(
            init=0.0,
            final=0.0,
        )
    )

    loss_depth: DepthLoss = field(default_factory=lambda: DepthLoss())

    lambda_feat: WeightScheduler = field(
        default_factory=lambda: WeightScheduler(
            init=0.0,
            final=0.0,
        )
    )

    def instantiate(self) -> "ImplicitMetricsImpl":
        return ImplicitMetricsImpl(self)


class ImplicitMetricsImpl(VanillaMetricsImpl):
    config: ImplicitMetrics

    _RENDER_TYPES = [
        "render",
        "alpha",
        "acc_depth",
        "inverse_depth",
        "normal",
        "unbiased_depth",
        "feature_map",
        "render_app",
        "feature_map_adapt",
    ]

    def setup(self, stage: str, pl_module: lightning.LightningModule) -> None:
        super().setup(stage, pl_module)

        self.config.lambda_depth.check(pl_module.trainer.max_steps)
        self.config.lambda_feat.check(pl_module.trainer.max_steps)

    def _get_basic_metrics(
        self,
        pl_module: lightning.LightningModule,
        gaussian_model: Union[ScaffoldGaussianModel, OctreeGaussianModel],
        batch: tuple,
        outputs: dict,
    ):
        global_step = pl_module.trainer.global_step + 1
        batch, outputs = self.batchify(batch, outputs)
        cameras, (image_names, gt_images, masks), extra_data = batch

        # basic metrics
        render, render_app = outputs["render"], outputs["render_app"]
        if masks is not None:
            gt_images = gt_images * masks
            render = render * masks
            if render_app is not None:
                render_app = render_app * masks
        rgb_diff_loss = self.rgb_diff_loss_fn(render_app or render, gt_images)
        ssim_metric = self.ssim(render, gt_images)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1.0 - ssim_metric)
        metrics = {"loss": loss, "rgb_diff": rgb_diff_loss, "ssim": ssim_metric}
        prog_bar = {"loss": True, "rgb_diff": True, "ssim": True}

        gaussians: NeuralGaussianWrapper = outputs["neural_gaussians"]
        if self.config.lambda_dreg > 0.0:
            dreg = torch.prod(gaussians.scales, dim=-1).mean()
            metrics["loss"] += self.config.lambda_dreg * dreg
            metrics["loss_dreg"] = dreg
            prog_bar["loss_dreg"] = False

        # align inverse depth
        weight_depth = self.config.lambda_depth(global_step)
        if weight_depth > 0.0:
            depth_list = extra_data.get(InverseDepthData._KEY, [])
            if not isinstance(depth_list, list):
                depth_list = [depth_list]
            valid_ids = [idx for idx, d in enumerate(depth_list) if d is not None]

            loss_depth = torch.tensor(0.0, device=metrics["loss"].device)
            if len(valid_ids) > 0:
                gt_depth = torch.stack([depth_list[idx] for idx in valid_ids], dim=0)
                depth = outputs["inverse_depth"][valid_ids].squeeze(1)
                if masks is not None:
                    gt_depth = gt_depth * masks[valid_ids]
                    depth = depth * masks[valid_ids]
                loss_depth = self.config.loss_depth(depth, gt_depth)

            metrics["loss"] += weight_depth * loss_depth
            metrics["loss_depth"] = loss_depth
            prog_bar["loss_depth"] = True

        # align semantic feature
        weight_feat = self.config.lambda_feat(global_step)
        if weight_feat > 0.0:
            feature_list = extra_data.get(FeatureData._KEY, [])
            if not isinstance(feature_list, list):
                feature_list = [feature_list]

            loss_feat = torch.tensor(0.0, device=metrics["loss"].device)
            if len(feature_list) > 0:
                gt_feat = torch.stack(feature_list, dim=0)
                feat, feat_adapt = outputs["feature_map"], outputs["feature_map_adapt"]

                if masks is not None:
                    masks_interp = F.interpolate(masks, size=feat.shape[-2:], mode="bilinear", align_corners=True)
                    gt_feat = gt_feat * masks_interp
                    if feat_adapt is not None:
                        feat_adapt = feat_adapt * masks_interp
                    else:
                        feat = feat * masks_interp
                loss_feat = F.l1_loss(feat_adapt or feat, gt_feat)

            metrics["loss"] += weight_feat * loss_feat
            metrics["loss_feat"] = loss_feat
            prog_bar["loss_feat"] = True

        # whether the feature of next step should be loaded
        load_next = self.config.lambda_feat(global_step + 1) > 0.0
        parser_outputs = pl_module.trainer.datamodule.dataparser_outputs
        for s in [parser_outputs.train_set, parser_outputs.val_set]:
            enabled = getattr(s.extra_data_processor, "enabled", None)
            if enabled is not None:
                s.extra_data_processor.enabled["feature"] = load_next

        return metrics, prog_bar

    def batchify(self, batch, outputs):
        cameras, (image_names, gt_images, masks), extra_data = batch
        if len(cameras.R.shape) == 2:
            cameras = BatchedCameras.batchify_cameras(cameras)
            image_names = [image_names]
            gt_images = gt_images.unsqueeze(0)
            masks = [masks]

            _outputs = {}
            for key, val in outputs.items():
                if key in self._RENDER_TYPES and isinstance(val, torch.Tensor):
                    val = val.unsqueeze(0)
                _outputs[key] = val
            outputs = _outputs

        if masks is None or all([m is None for m in masks]):
            masks = None
        else:
            masks = torch.stack([m or gt_images[i].new_ones(gt_images[i].shape) for i, m in enumerate(masks)], dim=0)

        batch = (cameras, (image_names, gt_images, masks), extra_data)

        return batch, outputs

    @staticmethod
    def _create_fused_ssim_adapter():
        # fmt: off
        from fused_ssim import fused_ssim
        def adapter(pred, gt):
            if len(pred.shape) == 3:
                pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
            return fused_ssim(pred, gt)
        # fmt: on
        return adapter
