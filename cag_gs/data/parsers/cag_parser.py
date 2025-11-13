import json
import math
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Camera, Cameras
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser

from ..dataset_utils import ExtraDataProcessor, FeatureData, InverseDepthData


@dataclass
class DepthConfig:
    dir: str = ""
    depth_rescaling: bool = True
    depth_scale_filename: str = "estimated_depth_scales.json"
    depth_scale_lower_bound: float = 0.2
    depth_scale_upper_bound: float = 5.0

    def configure(self, path: str, outputs: DataParserOutputs):
        if self.dir is None or len(self.dir) <= 0:
            return
        if self.depth_rescaling:
            with open(osp.abspath(osp.join(path, self.depth_scale_filename)), "r") as f:
                depth_scales = json.load(f)
            image_name_set = {n for n in outputs.train_set.image_names + outputs.val_set.image_names}
            depth_scale_list = []
            for image_name, image_depth_scale in depth_scales.items():
                if image_name not in image_name_set:
                    continue
                depth_scale_list.append(image_depth_scale["scale"])

            median_scale = np.median(np.asarray(depth_scale_list))

        for image_set in [outputs.train_set, outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = osp.abspath(osp.join(path, self.dir, f"{image_name}.npy"))
                if not osp.exists(depth_file_path):
                    depth_file_path = None
                depth_scale = {"scale": 1.0, "offset": 0.0}
                # if not satisfy requirements, set `depth_scale` to None
                if depth_file_path is not None and self.depth_rescaling:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is not None and (
                        depth_scale["scale"] < self.depth_scale_lower_bound * median_scale
                        or depth_scale["scale"] > self.depth_scale_upper_bound * median_scale
                    ):
                        depth_scale = None

                if depth_file_path is not None and depth_scale is not None:
                    data = InverseDepthData(
                        depth_file_path,
                        (image_set.cameras[idx].height.item(), image_set.cameras[idx].width.item()),
                        depth_scale["scale"],
                        depth_scale["offset"],
                    )
                else:
                    data = InverseDepthData()

                image_set.extra_data[idx][data._KEY] = data


@dataclass
class SemanticConfig:
    dir: str = ""

    def configure(self, path: str, dataparser_outputs: DataParserOutputs):
        if self.dir is None or len(self.dir) <= 0:
            return

        feature_dim = -1
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                filepath = osp.join(path, self.dir, f"{image_name}.npy")
                if osp.exists(filepath):
                    data = FeatureData(filepath)
                    if feature_dim < 0:
                        feature_dim = data._shape[-1]
                else:
                    data = FeatureData()

                image_set.extra_data[idx][data._KEY] = data


@dataclass
class CAGParser(Colmap):
    depth_config: DepthConfig = field(default_factory=lambda: DepthConfig())

    feature_config: SemanticConfig = field(default_factory=lambda: SemanticConfig())

    camera_extent_all_images: bool = False

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return CAGDataParser(path, output_path, global_rank, self)


class CAGDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: CAGParser):
        self.params: CAGParser
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        outputs = super().get_outputs()

        # set extra data and processor
        for image_set in [outputs.train_set, outputs.val_set]:
            if not isinstance(image_set.extra_data_processor, ExtraDataProcessor):
                image_set.extra_data_processor = ExtraDataProcessor()
            for idx in range(len(image_set.image_names)):
                if not isinstance(image_set.extra_data[idx], dict):
                    image_set.extra_data[idx] = {}

        self.params.depth_config.configure(self.path, outputs)
        self.params.feature_config.configure(self.path, outputs)

        return outputs
