import argparse
import csv
import json
import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from internal.dataparsers import DataParserOutputs
from internal.dataset import CacheDataLoader, Dataset
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.visualizers import Visualizers
from utils.common import AsyncImageSaver, AsyncNDArraySaver

# params for efficiency measurement
WARM_UP_ITERS = 100
N_TIMES = 20  # number of times to repeat the rendering for averaging
N_VAL_IMAGES = 300  # if valid cameras more than this, do not repeat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-c", type=str, required=True)
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="")
    parser.add_argument("--eval_efficiency", action="store_true")

    args = parser.parse_args()
    if not (len(args.output_dir) > 0):
        args.output_dir = osp.dirname(args.ckpt_path)
    return args


def load_from_ckpt(args, device):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)

    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    dataparser_config.points_from = "random"
    dataparser_outputs: DataParserOutputs = dataparser_config.instantiate(
        path=args.dataset_path, output_path=os.getcwd(), global_rank=0
    ).get_outputs()

    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, stage="validation", device=device)

    return gaussian_model, renderer, dataparser_outputs, ckpt


def get_image_saver(async_image_saver, output_dir: str):
    for d in ["render", "gt"]:
        os.makedirs(osp.join(output_dir, d), exist_ok=True)

    def tensor2numpy(image: torch.Tensor):
        return (image * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def _image_saver(batch, outputs):
        image_name = str(Path(batch[1][0]).with_suffix(".png"))

        render = tensor2numpy(torch.clamp_max(outputs["render"], max=1.0))
        gt = tensor2numpy(batch[1][1])
        # montage = np.concatenate([render, gt], axis=1)

        async_image_saver.save(render, osp.join(output_dir, "render/{}.png".format(image_name)))
        async_image_saver.save(gt, osp.join(output_dir, "gt/{}.png".format(image_name)))
        # async_image_saver.save(montage, osp.join(output_dir, "montage/{}.png".format(image_name)))

    return _image_saver


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ckpt
    gaussian_model, renderer, dataparser_outputs, ckpt = load_from_ckpt(args, device)
    dataloader = CacheDataLoader(
        Dataset(
            dataparser_outputs.val_set,
            undistort_image=False,
            camera_device=device,
            image_device=torch.device("cpu"),
        ),
        max_cache_num=-1,
        shuffle=False,
        num_workers=6,
    )

    # Create image saver
    async_image_saver = AsyncImageSaver(is_rgb=True)
    image_saver = get_image_saver(async_image_saver, args.output_dir)

    # Render
    bg_color = torch.tensor(ckpt["hyper_parameters"]["background_color"], dtype=torch.float32, device=device)
    for batch in tqdm(dataloader, desc="Rendering and evaluating"):
        camera, (name, image, mask), extra = batch
        image = image.to(device)
        outputs = renderer(camera, gaussian_model, bg_color)
        image_saver(batch, outputs)

    async_image_saver.stop()

    # Evaluate efficiency
    if args.eval_efficiency:
        # Warm up
        cnt = 0
        bg_color = torch.tensor(ckpt["hyper_parameters"]["background_color"], dtype=torch.float32, device=device)
        with tqdm(total=WARM_UP_ITERS, desc="Warming up") as pbar:
            for batch in dataloader:
                camera, (name, _, _), _ = batch
                outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb"])
                pbar.update(1)
                cnt += 1
                if cnt >= WARM_UP_ITERS:
                    break

        n_times = 1 if len(dataloader) > N_VAL_IMAGES else N_TIMES
        time_list = [{} for _ in range(n_times)]
        total_time, max_time = 0.0, 0.0

        torch.cuda.reset_peak_memory_stats(device)
        for idx in range(n_times):
            for batch in tqdm(dataloader, desc="Measuring efficiency"):
                camera, (name, image, mask), extra = batch
                # image = image.to(device)

                torch.cuda.synchronize()
                start_time = time.time()
                outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb"])
                torch.cuda.synchronize()
                end_time = time.time()

                elapsed_time = end_time - start_time
                time_list[idx][name] = elapsed_time
                total_time += elapsed_time
                max_time = max(max_time, elapsed_time)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        avg_time = total_time / (len(dataloader) * n_times)
        with open(osp.join(args.output_dir, "efficiency_summary.json"), "w") as f:
            summary = {
                "avg_time_ms": avg_time * 1000.0,
                "max_time_ms": max_time * 1000.0,
                "avg_fps": 1.0 / avg_time,
                "peak_memory_MiB": peak_mem,
            }
            json.dump(summary, f, indent=4, separators=(", ", ": "))
        with open(osp.join(args.output_dir, "efficiency.csv"), "w") as f:
            metrics_writer = csv.writer(f)
            metrics_writer.writerow(["name"] + [f"time_run_{i}" for i in range(n_times)])
            for name in time_list[0].keys():
                metric_row = [name]
                for i in range(n_times):
                    metric_row.append(time_list[i][name] * 1000.0)
                metrics_writer.writerow(metric_row)


if __name__ == "__main__":
    main()
