import argparse
import gc
import json
import os
import os.path as osp
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from cag_gs.modeling.models.cag_gaussian import ConsistentAnchorGuidedGaussianModel
from cag_gs.modeling.models.partitionable_gaussian import PartitionableOctreeGaussian, PartitionableOctreeGaussianModel
from cag_gs.modeling.utils.partitionable_scene import MinMaxBoundingBox, MinMaxBoundingBoxes, PartitionCoordinates
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import build_rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", type=str, required=True, help="Project Name")
    parser.add_argument("--output_path", "-o", type=str)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = osp.join("outputs", args.project, "merged")
    return args


def build_transform4x4(transforms: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(4, device=transforms.device)
    matrix[:3, :3] = build_rotation(transforms[:4].unsqueeze(0)).squeeze(0)
    matrix[:3, 3] = transforms[4:]
    return matrix


def unbound_bbox(bbox: MinMaxBoundingBox, cell_id: Tuple[int], dims: Tuple[int]):
    x_id, y_id = cell_id
    x_dim, y_dim = dims
    bbox_min, bbox_max = bbox.min.clone(), bbox.max.clone()
    if x_id == 0:
        bbox_min[0] = -torch.inf
    if x_id == x_dim - 1:
        bbox_max[0] = torch.inf
    if y_id == 0:
        bbox_min[1] = -torch.inf
    if y_id == y_dim - 1:
        bbox_max[1] = torch.inf
    return MinMaxBoundingBox(bbox_min, bbox_max)


def split_anchors(
    gaussian_model: ConsistentAnchorGuidedGaussianModel, bbox: MinMaxBoundingBox, transform: torch.Tensor
):
    anchors = gaussian_model.get_anchors
    offsets = gaussian_model.get_offsets
    offset_scalings = gaussian_model.get_scales[:, :3]
    n_anchors, n_offsets = offsets.shape[:2]
    gaussian_means = anchors.unsqueeze(1) + offset_scalings.unsqueeze(1) * offsets

    transform = transform.to(gaussian_means)
    gaussian_means_ = gaussian_means @ transform[:3, :3].T + transform[:3, -1]
    is_in_bbox = torch.logical_and(
        (gaussian_means_[..., :2] >= bbox.min[:2]).all(dim=-1), (gaussian_means_[..., :2] <= bbox.max[:2]).all(dim=-1)
    ).reshape(n_anchors, n_offsets)

    anchor_mask = is_in_bbox.any(dim=1)
    inside_part = {}
    for k, v in gaussian_model.properties.items():
        inside_part[k] = v[anchor_mask]
    gaussian_model.properties = inside_part

    all_in_block = is_in_bbox[anchor_mask]
    invalid_indices = torch.nonzero(~all_in_block)
    return gaussian_model, invalid_indices


def update_ckpt(ckpt, merged_gaussians, implicit_properties_to_merge):
    ckpt["state_dict"]["gaussian_model._anchor_start_ids_per_block"] = torch.tensor(
        implicit_properties_to_merge["anchor_start_ids_per_block"], dtype=torch.long
    )
    keys_to_remove = [
        k
        for k in ckpt["state_dict"]
        if k.startswith(
            ("gaussian_model.neural_decoder.", "gaussian_model.hash_grid.", "gaussian_model.feature_adapter.")
        )
    ]
    for k in keys_to_remove:
        del ckpt["state_dict"][k]
    neural_decoder = nn.ModuleList(implicit_properties_to_merge["neural_decoder"])
    ckpt["state_dict"].update({f"gaussian_model.neural_decoder.{k}": v for k, v in neural_decoder.state_dict().items()})
    orig_gaussian = ckpt["hyper_parameters"]["gaussian"]
    params = {
        k: getattr(orig_gaussian, k)
        for k, v in PartitionableOctreeGaussian.__dataclass_fields__.items()
        if k in orig_gaussian.__dict__ and v.init
    }
    ckpt["hyper_parameters"]["gaussian"] = PartitionableOctreeGaussian(**params)

    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians.") or i.startswith("frozen_gaussians."):
            del ckpt["state_dict"][i]
    ckpt["optimizer_states"] = []
    ckpt["lr_schedulers"] = []
    ckpt["epoch"] = 0
    ckpt["global_step"] = 0
    ckpt["loops"] = {}

    if isinstance(ckpt["hyper_parameters"]["density"], VanillaDensityController):
        for k in list(ckpt["state_dict"].keys()):
            if k.startswith("density_controller."):
                ckpt["state_dict"][k] = torch.zeros(
                    (merged_gaussians["means"].shape[0], *ckpt["state_dict"][k].shape[1:]),
                    dtype=ckpt["state_dict"][k].dtype,
                )
    for k, v in merged_gaussians.items():
        ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = v


def main():
    # fmt: off
    MERGABLE_PROPERTY_NAMES = [
        "means", "shs_dc", "shs_rest", "scales", "rotations",
        "offsets", "features", "levels", "extra_levels",
    ]
    # fmt: on

    args = parse_args()
    device = torch.device("cpu")
    torch.autograd.set_grad_enabled(False)

    project_dir = Path("outputs") / args.project
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    metadata = json.load(open(str((project_dir / "partition/metadata.json").absolute()), "r"))
    transform4x4 = build_transform4x4(torch.tensor(metadata["scene"]["transforms"], device=device, dtype=torch.float32))
    scene_bbox = metadata["scene"]["bbox"]
    scene_bbox = MinMaxBoundingBox(
        min=torch.tensor(scene_bbox[:2]).to(transform4x4),
        max=torch.tensor(scene_bbox[2:]).to(transform4x4),
    )
    # Build partition coordinates
    block_xys, block_sizes, block_ids = [], [], []
    for block in metadata["blocks"]:
        bbox = block["bbox"]
        block_xys.append(torch.tensor(bbox[:2]).to(transform4x4))
        block_sizes.append((torch.tensor(bbox[2:]) - torch.tensor(bbox[:2])).to(transform4x4))
        block_ids.append(torch.tensor(block["block_id"], dtype=torch.long, device=device))
    partition_coords = PartitionCoordinates(
        id=torch.stack(block_ids, 0), xy=torch.stack(block_xys, 0), size=torch.stack(block_sizes, 0)
    )
    x_dim, y_dim = partition_coords.id[:, 0].max().item() + 1, partition_coords.id[:, 1].max().item() + 1

    num_gaussians_merged = 0
    gaussians_to_merge = {}
    implicit_properties_to_merge = {}

    with tqdm(enumerate(partition_coords), desc="Merging blocks") as pbar:
        for idx, (block_id, block_xy, block_size) in pbar:
            name = metadata["blocks"][idx]["name"]
            pbar.set_description("{}".format(name))
            bbox = MinMaxBoundingBox(min=block_xy, max=block_xy + block_size)
            bbox_unbound = unbound_bbox(bbox, block_id.tolist(), (x_dim, y_dim))

            # Load checkpoint
            ckpt_path = GaussianModelLoader.search_load_file(str((project_dir / "blocks" / name).absolute()))
            pbar.write("Loading checkpoint of {}".format(name))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")

            # Split anchors
            gaussian_model, invalid_indices = split_anchors(gaussian_model, bbox_unbound, transform4x4)
            gaussian_model.to("cuda")
            anchors = gaussian_model.get_anchors
            anchor_ids = torch.arange(anchors.shape[0], device=anchors.device)
            features = gaussian_model.compute_features(None, anchor_ids)
            gaussian_model.set_property("features", features)
            # implicit_properties_to_merge.setdefault("anchor_partition_ids", {}).update(
            #     {name: torch.full((anchors.shape[0],), idx, dtype=torch.int)}
            # )
            # implicit_properties_to_merge.setdefault("invalid_gaussian_ids", {}).update({name: invalid_indices})
            implicit_properties_to_merge.setdefault("anchor_start_ids_per_block", []).append(num_gaussians_merged)
            implicit_properties_to_merge.setdefault("neural_decoder", []).append(gaussian_model.neural_decoder)
            num_gaussians_merged += anchor_ids.shape[0]

            # Merge properties
            for i in MERGABLE_PROPERTY_NAMES:
                if i in gaussian_model.property_names:
                    gaussians_to_merge.setdefault(i, []).append(gaussian_model.get_property(i))

        merged_gaussians = {}
        for k, v in gaussians_to_merge.items():
            merged_gaussians[k] = torch.cat(v, 0)
            v.clear()
            gc.collect()
            torch.cuda.empty_cache()

        update_ckpt(ckpt, merged_gaussians, implicit_properties_to_merge)

        # Save
        print("Saving...")
        torch.save(ckpt, str((output_path / "merged.ckpt").absolute()))
        print("Saved to '{}'".format(str(output_path / "merged.ckpt")))


if __name__ == "__main__":
    main()
