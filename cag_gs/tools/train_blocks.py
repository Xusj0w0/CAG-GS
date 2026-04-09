import argparse
import json
import os
import os.path as osp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import py3nvml.py3nvml as nvml

from utils.argparser_utils import parser_stoppable_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        required=True,
        help="Project name. Output path will be `outputs/{project}`",
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file.")
    parser.add_argument("--gpu_indices", type=str, default="0,1,2,3,4,5,6,7")
    args, training_args = parser_stoppable_args(parser)
    return args, training_args


def train_a_block(ckpt_path, project, block, config_path, training_args, free_gpu):
    block_idx, block_name = block
    gpu_id, gpu_uuid = free_gpu
    output_path = osp.join("outputs", project, "blocks")
    print("Training {}. Output directory: {}".format(block_name, output_path))

    # fmt: off
    args = [
        "python", "-u", "main.py", "fit",
        "--config={}".format(config_path),
        "--output={}".format(output_path),
        "--name={}".format(block_name),
        "--data.parser.image_list={}".format(osp.join("outputs", project, "partition", f"partitions/{block_name}", "image_list.txt")),
        "--model.initialize_from={}".format(ckpt_path),
        "--logger=tensorboard",
    ]
    # fmt: on
    args += training_args
    print(args)

    os.makedirs(os.path.join(output_path, block_name), exist_ok=True)
    with open(os.path.join(output_path, block_name, "command.txt"), "w") as f:
        f.write(" ".join(args) + "\n")
    with open(os.path.join(output_path, block_name, "train.log"), "w") as f:
        f.write(" ".join(args) + "\n")

        ret_code = subprocess.run(
            args,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TQDM_MININTERVAL": "5",
                "TQDM_MINITERS": "200",
            },
            stdout=f,
            stderr=f,
            text=True,
            bufsize=1,
        )
    if ret_code.returncode == 0:
        trained_blocks_dir = osp.join("outputs", project, "trained")
        os.makedirs(trained_blocks_dir, exist_ok=True)
        with open(os.path.join(trained_blocks_dir, f"{block_name}.txt"), "w") as f:
            f.write("Trained")


def get_free_gpus(
    valid_gpu_ids: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    max_vram_usage: int = 1000,
) -> List[int]:
    """
    Args:
        valid_gpu_ids (List): NVML GPU indices to consider (e.g., [0,1,2,3])
        max_vram_usage (int): in MiB
    Returns:
        available_gpus (List): List of GPU UUIDs that have less than `max_vram_usage` MB used VRAM.
    """
    MiB = 1024 * 1024
    threshold = max_vram_usage * MiB
    n = nvml.nvmlDeviceGetCount()
    valid_ids = [i for i in valid_gpu_ids if 0 <= i < n]
    available_gpus = []
    for i in valid_ids:
        h = nvml.nvmlDeviceGetHandleByIndex(i)
        mem = nvml.nvmlDeviceGetMemoryInfo(h)
        if mem.used < threshold:
            uuid = nvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            available_gpus.append((i, uuid))
    return available_gpus


def main():
    args, training_args = parse_args()

    assert Path(args.config).exists(), f"Config file {args.config} does not exist."
    gpu_indices = [int(i) for i in args.gpu_indices.split(",")]

    part_info_dir = osp.join("outputs", args.project, "partition")
    metadata = json.load(open(osp.join(part_info_dir, "metadata.json"), "r"))
    ckpt_path = metadata["checkpoint_path"]
    num_blocks = len(metadata["blocks"])

    nvml.nvmlInit()
    tasks = []
    with ProcessPoolExecutor(max_workers=num_blocks) as ex:
        for block_idx in range(num_blocks):
            name = metadata["blocks"][block_idx]["name"]
            gpu_available = False
            fail_cnt = 0
            while not gpu_available:
                free_gpus = get_free_gpus(valid_gpu_ids=gpu_indices)
                if len(free_gpus) > 0:
                    gpu_available = True
                elif fail_cnt >= 240:
                    print("No free GPUs available in 8 hour, exiting...")
                else:
                    fail_cnt += 1
                    print("No free GPUs available, waiting for 2 minutes...")
                    time.sleep(120)

            gpu_id, gpu_uuid = free_gpus[-1]
            tasks.append(
                ex.submit(
                    train_a_block,
                    ckpt_path,
                    args.project,
                    (block_idx, name),
                    args.config,
                    training_args,
                    (gpu_id, gpu_uuid),
                )
            )
            time.sleep(120)

        for f in as_completed(tasks):
            f.result()


if __name__ == "__main__":
    main()
