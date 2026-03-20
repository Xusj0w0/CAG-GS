import warnings

warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter.*",
    category=UserWarning,
)

import argparse
import gc
import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
# from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from internal.utils.visualizers import Visualizers
from utils.common import AsyncImageReader, AsyncImageSaver, AsyncNDArraySaver
from utils.distibuted_tasks import (configure_arg_parser,
                                    get_task_list_with_args)

CFGs = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def make_parser():
    parser = argparse.ArgumentParser(description="Extract HWC feature to npy file")
    parser.add_argument("image_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sam2_ckpt", type=str, default="checkpoints/sam2/sam2.1_hiera_large.pt")
    parser.add_argument(
        "--feat_dim",
        type=str,
        default="32",
        help="should in [32, 64, 256], corresponding shape are [32x256x256, 64x128x128, 256x64x64]",
    )
    parser.add_argument("--preview", action="store_true", default=False)
    parser.add_argument("--ext", "-e", nargs="+", default=["jpg", "jpeg", "JPG", "JPEG", "png"])
    configure_arg_parser(parser)
    return parser


def parse_arch(ckpt_path):
    arch = [k for k in CFGs if k in os.path.basename(ckpt_path)]
    if len(arch) == 0:
        raise ValueError(f"Cannot find arch in {ckpt_path}, should be one of {list(CFGs.keys())}")
    return arch[0]


def build_sam_predictor(ckpt_path, arch):
    sam2 = build_sam2(CFGs[arch], ckpt_path, device=MODEL_DEVICE)
    predictor = SAM2ImagePredictor(sam2)
    return predictor


def get_image_list(args):
    image_name_set = {}
    for e in args.ext:
        for i in glob(os.path.join(image_path, f"**/*.{e}"), recursive=True):
            image_name_set[i] = True
    image_list = list(image_name_set.keys())
    assert len(image_list) > 0, "Not a image can be found"
    print(f"{len(image_list)} images found")
    image_list.sort()

    image_list = get_task_list_with_args(args, image_list)
    return image_list


if __name__ == "__main__":
    args = make_parser().parse_args()
    assert args.feat_dim in ["32", "64", "256"], "feat_dim should be in [32, 64, 256]"

    MODEL_DEVICE = "cuda"

    # build paths
    image_path = args.image_path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path.rstrip("/")))
    print(f"output_path={os.path.join(output_path, 'extra')}")

    # build output dirs
    arch = parse_arch(args.sam2_ckpt)
    feature_dir = os.path.join(output_path, "extra", "sam2_{}_d{}".format(arch, args.feat_dim))
    feature_preview_dir = os.path.join(output_path, "extra", "sam2_{}_d{}_preview".format(arch, args.feat_dim))
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(feature_preview_dir, exist_ok=True)

    # initialize SAM
    print("Initializing SAM...")
    predictor = build_sam_predictor(args.sam2_ckpt, arch)

    print("Finding image files...")
    image_list = get_image_list(args)

    image_reader = AsyncImageReader(image_list)
    image_saver = AsyncImageSaver()
    ndarray_saver = AsyncNDArraySaver()

    try:
        with tqdm(range(len(image_list))) as t:
            for _ in t:
                # get image information
                image_full_path, img = image_reader.queue.get()
                image_name = image_full_path[len(image_path) :].lstrip("/")

                t.set_description(f"{image_name}")
                semantic_file_name = f"{image_name}.npy"

                # predict image embedding
                min_size = min(*img.shape[:2])
                img = cv2.resize(img, dsize=(min_size, min_size), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                predictor.set_image(img)

                if args.feat_dim == "256":
                    image_embedding = predictor._features["image_embed"]  # [C, H, W]
                elif args.feat_dim == "64":
                    image_embedding = predictor._features["high_res_feats"][1]
                elif args.feat_dim == "32":
                    image_embedding = predictor._features["high_res_feats"][0]
                else:
                    raise ValueError(f"feat_dim {args.feat_dim} not supported")
                image_embedding = image_embedding.squeeze()

                # save
                image_embedding_np = image_embedding.permute(1, 2, 0).clone().detach().cpu().numpy()
                ndarray_saver.save(image_embedding_np, os.path.join(feature_dir, semantic_file_name))

                # preview
                if args.preview is True:
                    # image_embedding_flatten_normalized = torch.nn.functional.normalize(
                    #     image_embedding.permute(1, 2, 0).reshape((-1, image_embedding.shape[0])), dim=-1
                    # )
                    # pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(
                    #     semantic_features=image_embedding_flatten_normalized
                    # )
                    # pca_color = SegAnyGSUtils.get_pca_projected_colors(
                    #     image_embedding_flatten_normalized, pca_projection_matrix
                    # )
                    # feature_preview = pca_color.reshape((image_embedding.shape[1], image_embedding.shape[2], -1))
                    # feature_preview = (feature_preview * 255).to(torch.uint8).cpu().numpy()
                    # image_saver.save(feature_preview, os.path.join(feature_preview_dir, f"{image_name}.png"))
                    feature_preview = Visualizers.pca_colormap(torch.nn.functional.normalize(image_embedding, dim=0))
                    torchvision.utils.save_image(
                        feature_preview, os.path.join(feature_preview_dir, f"{image_name}.png")
                    )
                    # feature_preview = feature_preview.permute(1, 2, 0).clone().detach().cpu().numpy()
                    # feature_preview = (feature_preview * 255).astype(np.uint8)
                    # image_saver.save(feature_preview, os.path.join(feature_preview_dir, f"{image_name}.png"))
    finally:
        image_reader.stop()
        image_saver.stop()
        ndarray_saver.stop()
