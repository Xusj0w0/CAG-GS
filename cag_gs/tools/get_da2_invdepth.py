import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from external.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from internal.utils.visualizers import Visualizers
from utils.common import (AsyncImageReader, AsyncImageSaver, AsyncNDArraySaver,
                          AsyncTensorSaver, find_files)
from utils.distibuted_tasks import (configure_arg_parser,
                                    get_task_list_with_args)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--input_size", "-s", type=int, default=518)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--encoder", default="vitl")
    parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
    parser.add_argument("--da2_ckpt", "-c", type=str, default="checkpoints/da2/depth_anything_v2_vitl.pth")
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--colormap", type=str, default="gray")
    configure_arg_parser(parser)
    return parser


def build_da2_backbone(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.da2_ckpt, map_location="cpu"))
    depth_anything = depth_anything.to(DEVICE).eval()

    return depth_anything


def apply_color_map(normalized_depth, colormap):
    colored_depth = Visualizers.float_colormap(torch.from_numpy(normalized_depth).unsqueeze(0), colormap=colormap)
    colored_depth = (colored_depth.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return colored_depth


if __name__ == "__main__":
    args = make_parser().parse_args()

    image_path = args.image_dir
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))
    print(f"output_path={output_path}")
    depth_dir = os.path.join(output_path, "estimated_invdepth")
    depth_preview_dir = os.path.join(output_path, "estimated_invdepth_preview")

    os.makedirs(depth_dir, exist_ok=True)
    if args.preview:
        os.makedirs(depth_preview_dir, exist_ok=True)

    images = get_task_list_with_args(args, find_files(args.image_dir, args.extensions, as_relative_path=False))
    assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(
        args.extensions, args.image_dir
    )

    depth_anything = build_da2_backbone(args)

    image_reader = AsyncImageReader(image_list=images)
    ndarray_saver = AsyncNDArraySaver()
    image_saver = AsyncImageSaver(is_rgb=True)

    try:
        with torch.no_grad(), tqdm(range(len(images))) as t:
            for _ in t:
                image_path, raw_image = image_reader.get()
                image_name = image_path[len(args.image_dir) :].lstrip(os.path.sep)
                output_filename = os.path.join(output_path, "{}.npy".format(image_name))

                depth = depth_anything.infer_image(raw_image, args.input_size)  # [H, W]
                normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

                ndarray_saver.save(normalized_depth, os.path.join(depth_dir, "{}.npy".format(image_name)))

                if args.preview:
                    image_saver.save(
                        normalized_depth,
                        os.path.join(depth_preview_dir, "{}.png".format(image_name)),
                        processor=lambda x: apply_color_map(x, colormap=args.colormap),
                    )

    finally:
        ndarray_saver.stop()
        image_reader.stop()
        image_saver.stop()
