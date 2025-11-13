import os
import os.path as osp
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Tuple, Union

import numpy as np
import torch


@dataclass
class NumpyData:
    path: str = ""
    _shape: Tuple[int] = field(init=False)

    _KEY: ClassVar[str] = "base"

    def __post_init__(self):
        _shape = self.parse_shape()
        if _shape is not None:
            self._shape = tuple(_shape)

    def parse_shape(self) -> List[int]:
        if not (osp.exists(self.path) and self.path.endswith(".npy")):
            return None

        with open(self.path, "rb") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError("Not a valid .npy file")

            major, minor = np.frombuffer(f.read(2), dtype=np.uint8)
            if major == 1:
                header_len = np.frombuffer(f.read(2), dtype=np.uint16)[0]
            elif major == 2:
                header_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            else:
                raise ValueError("Unsupported .npy version")

            header = f.read(header_len).decode("latin1")
            header_dict = eval(header)
        self._shape = header_dict["shape"]
        return self._shape

    def load_data(self) -> torch.Tensor:
        if osp.exists(self.path) and self.path.endswith(".npy"):
            data = torch.from_numpy(np.load(self.path))
            data = self._process(data)
            return data
        return None

    def _process(self, data: torch.Tensor) -> torch.Tensor:
        return data


@dataclass
class InverseDepthData(NumpyData):
    shape: Tuple[int] = field(default_factory=lambda: (-1, -1))
    scale: float = 1.0
    offset: float = 0.0

    _KEY: ClassVar[str] = "inverse_depth"

    def _process(self, data: torch.Tensor) -> torch.Tensor:
        data = data * self.scale + self.offset
        data = torch.clamp_min(data, min=0.0)
        if data.shape != self.shape:
            data = torch.nn.functional.interpolate(
                data[None, None, ...], size=self.shape, mode="bilinear", align_corners=True  # fmt: skip
            )[0, 0]
        return data


@dataclass
class FeatureData(NumpyData):
    _KEY: ClassVar[str] = "feature"


class ExtraDataProcessor:
    enabled: Dict[str, bool] = defaultdict(lambda: True)

    def __call__(self, extra_data: dict):
        results = {}
        for key in extra_data.keys():
            if self.enabled[key]:
                results[key] = extra_data[key].load_data()
            else:
                results[key] = None
        return results
