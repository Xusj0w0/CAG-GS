# modified from nerf-studio (https://github.com/nerfstudio-project/nerfstudio)
from abc import abstractmethod
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn

from internal.utils.sh_utils import eval_sh

from .base import TCNN_EXISTS, FieldComponent, tcnn

MAX_SH_DEGREE = 4


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        in_dim: int = 3,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:
        super().__init__(in_dim=in_dim)
        self.num_levels = num_levels
        self.min_res = min_res
        self.in_dim = in_dim
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * self.growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.hash_table = torch.empty(0)
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("HashEncoding")
            self.build_nn_modules()
        elif implementation == "tcnn":
            encoding_config = self.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                interpolation=interpolation,
            )
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=self.in_dim,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the hash encoding."""
        self.hash_table = torch.rand(size=(self.hash_table_size * self.num_levels, self.features_per_level)) * 2 - 1
        self.hash_table *= self.hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

    @classmethod
    def get_tcnn_encoding_config(
        cls, num_levels, features_per_level, log2_hashmap_size, min_res, growth_factor, interpolation=None
    ) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if interpolation is not None:
            encoding_config["interpolation"] = interpolation
        return encoding_config

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class MixedHashEncoding(Encoding):
    def __init__(
        self,
        num_levels: int = 12,
        min_res: int = 16,
        max_res: int = 512,
        log2_hashmap_size: int = 19,
        features_per_level: int = 4,
        num_levels_2d: int = 4,
        min_res_2d: int = 128,
        max_res_2d: int = 1024,
        log2_hashmap_size_2d: int = 15,
        features_per_level_2d: int = 4,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ):
        super().__init__(in_dim=3)
        self.num_levels, self.num_levels_2d = num_levels, num_levels_2d
        self.min_res, self.max_res = min_res, max_res
        self.min_res_2d, self.max_res_2d = min_res_2d, max_res_2d
        self.features_per_level, self.features_per_level_2d = features_per_level, features_per_level_2d
        self.log2_hashmap_size, self.log2_hashmap_size_2d = log2_hashmap_size, log2_hashmap_size_2d
        self.hash_init_scale = hash_init_scale

        self.implementation = implementation
        self.interpolation = interpolation

        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.growth_factor_2d = (
            np.exp((np.log(max_res_2d) - np.log(min_res_2d)) / (num_levels_2d - 1)) if num_levels_2d > 1 else 1
        )

        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            self.build_nn_modules()
        elif implementation == "tcnn":
            encoding_config = self.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                num_levels_2d=self.num_levels_2d,
                features_per_level_2d=self.features_per_level_2d,
                log2_hashmap_size_2d=self.log2_hashmap_size_2d,
                min_res_2d=self.min_res_2d,
                growth_factor_2d=self.growth_factor_2d,
                interpolation=interpolation,
            )
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=9,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def build_nn_modules(self) -> None:
        encoding_params = {
            "num_levels": self.num_levels,
            "min_res": self.min_res,
            "max_res": self.max_res,
            "log2_hashmap_size": self.log2_hashmap_size,
            "features_per_level": self.features_per_level,
            "hash_init_scale": self.hash_init_scale,
            "implementation": self.implementation,
            "interpolation": self.interpolation,
        }

        encoding_params_2d = {
            "num_levels": self.num_levels_2d,
            "min_res": self.min_res_2d,
            "max_res": self.max_res_2d,
            "log2_hashmap_size": self.log2_hashmap_size_2d,
            "features_per_level": self.features_per_level_2d,
            "hash_init_scale": self.hash_init_scale,
            "implementation": self.implementation,
            "interpolation": self.interpolation,
        }

        self.encodings = nn.ModuleDict(
            {
                "xyz": HashEncoding(in_dim=3, **encoding_params),
                "xy": HashEncoding(in_dim=2, **encoding_params_2d),
                "yz": HashEncoding(in_dim=2, **encoding_params_2d),
                "zx": HashEncoding(in_dim=2, **encoding_params_2d),
            }
        )

    @classmethod
    def get_tcnn_encoding_config(
        cls,
        num_levels,
        features_per_level,
        log2_hashmap_size,
        min_res,
        growth_factor,
        num_levels_2d,
        features_per_level_2d,
        log2_hashmap_size_2d,
        min_res_2d,
        growth_factor_2d,
        interpolation=None,
    ) -> dict:
        encoding_config = HashEncoding.get_tcnn_encoding_config(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            min_res=min_res,
            growth_factor=growth_factor,
            interpolation=interpolation,
        )
        encoding_config_2d = HashEncoding.get_tcnn_encoding_config(
            num_levels=num_levels_2d,
            features_per_level=features_per_level_2d,
            log2_hashmap_size=log2_hashmap_size_2d,
            min_res=min_res_2d,
            growth_factor=growth_factor_2d,
            interpolation=interpolation,
        )
        config = {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 3, **encoding_config},
                {"n_dims_to_encode": 2, **encoding_config_2d},
                {"n_dims_to_encode": 2, **encoding_config_2d},
                {"n_dims_to_encode": 2, **encoding_config_2d},
            ],
        }
        return config

    def get_out_dim(self):
        return self.num_levels * self.features_per_level + 3 * self.num_levels_2d * self.features_per_level_2d

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        xyz, xy, yz, zx = torch.split(in_tensor, [3, 2, 2, 2], dim=-1)
        xyz = self.encodings["xyz"](xyz)
        xy = self.encodings["xy"](xy)
        yz = self.encodings["yz"](yz)
        zx = self.encodings["zx"](zx)
        out = torch.cat([xyz, xy, yz, zx], dim=-1)
        return out

    @classmethod
    def reorganize_input(cls, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        assert in_tensor.shape[-1] == 3
        x, y, z = torch.chunk(in_tensor, 3, dim=-1)
        in_tensor = torch.cat([x, y, z, x, y, y, z, z, x], dim=-1)
        return in_tensor

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        in_tensor = self.reorganize_input(in_tensor)
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode. (level = sh degree + 1)
    """

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch") -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > MAX_SH_DEGREE + 1:
            raise ValueError(
                f"Spherical harmonic encoding only supports 1 to {MAX_SH_DEGREE + 1} levels, requested {levels}"
            )

        self.levels = levels

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("SHEncoding")
            pass
        elif implementation == "tcnn":
            encoding_config = self.get_tcnn_encoding_config(levels=self.levels)
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

    @classmethod
    def get_tcnn_encoding_config(cls, levels: int) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        encoding_config = {
            "otype": "SphericalHarmonics",
            "degree": levels,
        }
        return encoding_config

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return self.components_from_spherical_harmonics(degree=self.levels - 1, directions=in_tensor)

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)

    @classmethod
    def components_from_spherical_harmonics(
        cls, degree: int, directions: Float[Tensor, "*batch 3"]
    ) -> Float[Tensor, "*batch components"]:
        """
        Returns value for each component of spherical harmonics.

        Args:
            degree: Number of spherical harmonic degrees to compute.
            directions: Spherical harmonic coefficients
        """
        assert 0 <= degree <= MAX_SH_DEGREE, f"SH degree must be in [0, {MAX_SH_DEGREE}], got {degree}"
        assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

        num_components = (degree + 1) ** 2
        components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]

        xx = x**2
        yy = y**2
        zz = z**2

        # l0
        components[..., 0] = 0.28209479177387814

        # l1
        if degree > 0:
            components[..., 1] = 0.4886025119029199 * y
            components[..., 2] = 0.4886025119029199 * z
            components[..., 3] = 0.4886025119029199 * x

        # l2
        if degree > 1:
            components[..., 4] = 1.0925484305920792 * x * y
            components[..., 5] = 1.0925484305920792 * y * z
            components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
            components[..., 7] = 1.0925484305920792 * x * z
            components[..., 8] = 0.5462742152960396 * (xx - yy)

        # l3
        if degree > 2:
            components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
            components[..., 10] = 2.890611442640554 * x * y * z
            components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
            components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
            components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
            components[..., 14] = 1.445305721320277 * z * (xx - yy)
            components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

        # l4
        if degree > 3:
            components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
            components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
            components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
            components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
            components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
            components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
            components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
            components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
            components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return components
