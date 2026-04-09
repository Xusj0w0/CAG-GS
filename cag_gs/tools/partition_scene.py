import argparse
import os
import os.path as osp
from dataclasses import fields
from typing import Any, Dict

import torch

from cag_gs.modeling.utils.partitionable_scene import (PartitionableScene,
                                                       SceneConfig)
from internal.utils.gaussian_model_loader import GaussianModelLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", type=str, required=True)
    parser.add_argument("--coarse_model_path", "-c", type=str)
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--partition_dim", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--scene_bbox_enlarge_by_pts", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--scene_bbox_outlier_by_pts", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--scene_bbox_enlarge_by_campos", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bbox_enlarge_by_campos", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bbox_enlarge_by_camvis", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--camera_visibility_threshold", type=float, default=argparse.SUPPRESS)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    coarse_model_path = args.coarse_model_path or osp.join("outputs", args.project, "coarse")
    assert osp.exists(coarse_model_path), f"Coarse model path {coarse_model_path} does not exist"
    coarse_model_path = GaussianModelLoader.search_load_file(coarse_model_path)
    assert coarse_model_path.endswith(".ckpt"), "Only .ckpt coarse model is supported"
    output_path = osp.join("outputs", args.project, "partition")
    os.makedirs(output_path, exist_ok=True)

    args_dict = vars(args)
    params = {}
    for field in fields(SceneConfig):
        if field.name in args_dict:
            params[field.name] = args_dict[field.name]
    # override
    ckpt = torch.load(coarse_model_path, map_location="cpu")
    params.update(
        {
            "coarse_model_path": coarse_model_path,
            "transforms": ckpt["state_dict"]["gaussian_model.voxel_grid.transform"].tolist(),
        }
    )
    config = SceneConfig(**params)
    scene = config.instantiate()
    scene.run(output_path)


if __name__ == "__main__":
    main()
