import argparse
import csv
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, required=True)
    parser.add_argument("--gt", type=str, default="")
    parser.add_argument("--output", "-o", type=str, default="")

    args = parser.parse_args()
    if len(args.gt) == 0:
        args.gt = osp.join(osp.dirname(args.render), "gt")
    if len(args.output) == 0:
        args.output = osp.dirname(osp.dirname(args.render))
    return args


def get_metric_calculator(device):
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device=device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
    vgg_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device=device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device=device)

    def _metric_calculator(render, gt, mask=None):
        render = torch.clamp_max(render, max=1.0)
        gt = gt.to(render)

        # mask
        if mask is not None:
            render = render * mask
            gt = gt * mask

        return {
            "psnr": psnr(render, gt).squeeze(),
            "ssim": ssim(render, gt).squeeze(),
            "vgg_lpips": vgg_lpips(render, gt).squeeze(),
            "alex_lpips": alex_lpips(render, gt).squeeze(),
        }

    return _metric_calculator


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.output, exist_ok=True)

    metric_keys = ["psnr", "ssim", "vgg_lpips", "alex_lpips"]
    metric_list = {}
    metrics_per_image = {}
    metric_calculator = get_metric_calculator(device)

    for render_path in Path(args.render).glob("*"):
        gt_path = Path(args.gt) / render_path.name

        render = Image.open(str(render_path.absolute()))
        gt = Image.open(str(gt_path.absolute()))
        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device)
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device)

        metrics_per_image[render_path.name] = metric_calculator(render_tensor, gt_tensor)

    # Save metrics
    metric_keys = ["psnr", "ssim", "vgg_lpips", "alex_lpips"]
    metric_list = {}
    with open(osp.join(args.output, "quality.csv"), "w") as f:
        metrics_writer = csv.writer(f)
        metrics_writer.writerow(["name"] + metric_keys)
        for name, metrics in metrics_per_image.items():
            metric_row = [name]
            for k in metric_keys:
                v = metrics[k]
                metric_row.append(v.item())
                metric_list.setdefault(k, []).append(v)
            metrics_writer.writerow(metric_row)
        metrics_writer.writerow([""])
    with open(osp.join(args.output, "quality_summary.json"), "w") as f:
        summary = {k: torch.mean(torch.stack(metric_list[k]).float()).item() for k in metric_keys}
        json.dump(summary, f, indent=4, separators=(", ", ": "))
        print(summary)


if __name__ == "__main__":
    main()
