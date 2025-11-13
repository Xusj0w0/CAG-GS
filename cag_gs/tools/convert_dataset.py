import argparse
import concurrent.futures
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

from internal.utils.colmap import (Camera, Image, Point3D, read_model,
                                   write_model)


def make_parser():
    parser = argparse.ArgumentParser(description="Convert CityGS-format dataset to Gaussian Splatting Lightning")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--down_sample_factor", type=int, default=1)
    parser.add_argument("--rescale_width", type=int, default=-1)
    parser.add_argument("--ext", nargs="+", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
    parser.add_argument("--skip_image", default=False, action="store_true")
    parser.add_argument("--skip_sparse", default=False, action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    return parser


def load_sparse_model(sparse_model_path):
    sparse_model = read_model(sparse_model_path)
    if sparse_model is None:
        sparse_model = read_model(osp.join(sparse_model_path, "0"))
    if sparse_model is None:
        raise FileNotFoundError("Sparse model not founded.")

    return sparse_model


def merge_sparse_model(
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    points3d: Dict[int, Point3D],
    _cameras: Dict[int, Camera],
    _images: Dict[int, Image],
    _points3d: Dict[int, Point3D],
    prefix: str = "",
):
    def is_same_camera(camera: Camera, camera_ref: Camera):
        if camera.model != camera_ref.model:
            return False
        if camera.width != camera_ref.width or camera.height != camera_ref.height:
            return False
        if not np.allclose(camera.params, camera_ref.params, atol=1e-3):
            return False
        return True

    # merge cameras
    camera_id = max(cameras.keys())
    val_cam_mapping = {}
    for k, v in _cameras.items():
        same_cam_id = -1
        for ref_id, v_ref in cameras.items():
            if is_same_camera(v, v_ref):
                same_cam_id = ref_id
                break
        if same_cam_id < 0:
            camera_id += 1
            new_cam = Camera(id=camera_id, model=v.model, width=v.width, height=v.height, params=v.params)
            cameras[camera_id] = new_cam
            val_cam_mapping[k] = camera_id
        else:
            val_cam_mapping[k] = same_cam_id

    # merge images
    images = {
        k: Image(
            id=v.id,
            qvec=v.qvec,
            tvec=v.tvec,
            camera_id=v.camera_id,
            name=str(Path(v.name).with_suffix(".png")),
            xys=v.xys,
            point3D_ids=v.point3D_ids,
        )
        for k, v in images.items()
    }
    image_id = max(images.keys())
    val_img_mapping = {}
    for k, v in _images.items():  # "id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"
        image_id += 1
        new_img = Image(
            id=image_id,
            qvec=v.qvec,
            tvec=v.tvec,
            camera_id=val_cam_mapping[v.camera_id],
            name=str(add_prefix(Path(v.name), prefix).with_suffix(".png")),
            xys=v.xys,
            point3D_ids=np.empty((0,)),  # pts3d not merged yet
        )
        images[image_id] = new_img
        val_img_mapping[k] = image_id

    # merge points3d
    point3d_id = max(points3d.keys())
    val_pts3d_mapping = {}
    for k, v in _points3d.items():
        point3d_id += 1
        image_ids = np.array([val_img_mapping[i] for i in v.image_ids.tolist()])
        image_ids = image_ids.astype(v.image_ids.dtype)
        new_point = Point3D(
            id=point3d_id, xyz=v.xyz, rgb=v.rgb, error=v.error, image_ids=image_ids, point2D_idxs=v.point2D_idxs
        )
        points3d[point3d_id] = new_point
        val_pts3d_mapping[k] = point3d_id

    # update points3D_ids in images
    for img_id_orig, img_id_mapped in val_img_mapping.items():
        image_orig = _images[img_id_orig]
        image = images[img_id_mapped]
        point3d_ids = np.array([val_pts3d_mapping[i] if i > 0 else -1 for i in image_orig.point3D_ids.tolist()])
        point3d_ids = point3d_ids.astype(image_orig.point3D_ids.dtype)
        images[img_id_mapped] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=point3d_ids,
        )

    return cameras, images, points3d


def find_images(path: str, extensions: list) -> list:
    image_list = []
    for extension in extensions:
        image_list += Path(path).rglob("*.{}".format(extension))
    return image_list


def get_resized_size(width, height, down_sample_factor: int = 1, rescale_width: int = -1):
    if down_sample_factor > 0:
        resized_width, resized_height = round(width / down_sample_factor), round(height / down_sample_factor)
    elif rescale_width > 0:
        if width > height:
            resized_width = int(rescale_width)
            resized_height = round(float(height) / width * rescale_width)
        else:
            resized_height = int(rescale_width)
            resized_width = round(float(width) / height * rescale_width)
    else:
        raise NotImplementedError
    return resized_width, resized_height


def resize_image(image_path: str, dst_path: str, down_sample_factor: int = 1, rescale_width: int = -1):
    image = PILImage.open(image_path)

    width, height = image.size
    resized_width, resized_height = get_resized_size(width, height, down_sample_factor, rescale_width)

    resized_image = image.resize((resized_width, resized_height))

    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    resized_image.save(dst_path, quality=100)


def add_prefix(image_path: Path, prefix: str = "") -> Path:
    return Path(image_path.parent / (prefix + image_path.name))


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, "image_info"), "w") as f:
        if args.rescale_width > 0:
            f.write(f"Rescale long edge of images to {args.rescale_width} pixels")
            args.down_sample_factor = -1
        else:
            f.write(f"Down sample images by {args.down_sample_factor} times")

    # preprocess images
    if not args.skip_image:
        for split in ["train", "val"]:
            src_dir = osp.join(args.input_dir, split, "images")
            dst_dir = osp.join(args.output_dir, "images")
            image_list: List[Path] = find_images(src_dir, args.ext)
            with ThreadPoolExecutor() as tpe:
                future_list = []
                for src_path in image_list:
                    dst_path = Path(dst_dir) / src_path.relative_to(src_dir)
                    dst_path = add_prefix(dst_path, args.prefix if split == "val" else "").with_suffix(".png")
                    future_list.append(
                        tpe.submit(
                            resize_image,
                            str(src_path.absolute()),
                            str(dst_path.absolute()),
                            args.down_sample_factor,
                            args.rescale_width,
                        )
                    )

                for _ in tqdm(concurrent.futures.as_completed(future_list), total=len(future_list)):
                    pass

    # merge sparse model
    if not args.skip_sparse:
        train_cameras, train_images, train_points3d = load_sparse_model(osp.join(args.input_dir, "train/sparse"))
        val_cameras, val_images, val_points3d = load_sparse_model(osp.join(args.input_dir, "val/sparse"))
        cameras, images, points3d = merge_sparse_model(
            train_cameras, train_images, train_points3d, val_cameras, val_images, val_points3d, args.prefix
        )
        val_image_names = [v.name for v in val_images.values()]
        with open(osp.join(args.output_dir, "val_images.txt"), "w") as f:
            for image_name in val_image_names:
                f.write(str(add_prefix(Path(image_name)).with_suffix(".png")) + "\n")

        # rescaling images
        camera_scalings, _cameras = {}, {}
        for camera_id, camera in cameras.items():
            resized_width, resized_height = get_resized_size(
                camera.width, camera.height, args.down_sample_factor, args.rescale_width
            )
            scaling_x, scaling_y = float(resized_width) / camera.width, float(resized_height) / camera.height
            camera_scalings[camera.id] = (scaling_x, scaling_y)

            _params = deepcopy(camera.params)
            _params[0] *= scaling_x
            _params[2] *= scaling_x
            _params[1] *= scaling_y
            _params[3] *= scaling_y
            _cameras[camera_id] = Camera(
                id=camera.id,
                model=camera.model,
                width=resized_width,
                height=resized_height,
                params=_params,
            )
        _images = {}
        for image_id, image in images.items():
            _xys = image.xys
            _xys[:, 0] *= camera_scalings[image.camera_id][0]
            _xys[:, 1] *= camera_scalings[image.camera_id][1]
            _images[image_id] = Image(
                id=image.id,
                qvec=image.qvec,
                tvec=image.tvec,
                camera_id=image.camera_id,
                name=image.name,
                xys=_xys,
                point3D_ids=image.point3D_ids,
            )
        cameras, images = _cameras, _images

        # write model
        sparse_model_path = osp.join(args.output_dir, "sparse")
        os.makedirs(sparse_model_path, exist_ok=True)
        write_model(cameras, images, points3d, sparse_model_path)


if __name__ == "__main__":
    main()
