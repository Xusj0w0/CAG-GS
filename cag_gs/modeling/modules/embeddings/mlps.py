# modified from nerf-studio (https://github.com/nerfstudio-project/nerfstudio)
from typing import Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from .base import TCNN_EXISTS, FieldComponent, tcnn
from .encodings import HashEncoding, MixedHashEncoding, SHEncoding


def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


class MLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("MLP")
            self.build_nn_modules()
        elif implementation == "tcnn":
            network_config = self.get_tcnn_network_config(
                activation=self.activation,
                out_activation=self.out_activation,
                layer_width=self.layer_width,
                num_layers=self.num_layers,
            )
            self.tcnn_encoding = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=self.out_dim,
                network_config=network_config,
            )

    @classmethod
    def get_tcnn_network_config(cls, activation, out_activation, layer_width, num_layers) -> dict:
        """Get the network configuration for tcnn if implemented"""
        activation_str = activation_to_tcnn_string(activation)
        output_activation_str = activation_to_tcnn_string(out_activation)
        if layer_width in [16, 32, 64, 128]:
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": activation_str,
                "output_activation": output_activation_str,
                "n_neurons": layer_width,
                "n_hidden_layers": num_layers - 1,
            }
        else:
            network_config = {
                "otype": "CutlassMLP",
                "activation": activation_str,
                "output_activation": output_activation_str,
                "n_neurons": layer_width,
                "n_hidden_layers": num_layers - 1,
            }
        return network_config

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class MLPWithHashEncoding(FieldComponent):
    """Multilayer perceptron with hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = 3

        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("MLPWithHashEncoding")
            self.build_nn_modules()
        elif implementation == "tcnn":
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.in_dim,
                n_output_dims=self.out_dim,
                encoding_config=HashEncoding.get_tcnn_encoding_config(
                    num_levels=self.num_levels,
                    features_per_level=self.features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    min_res=self.min_res,
                    growth_factor=self.growth_factor,
                    interpolation=interpolation,
                ),
                network_config=MLP.get_tcnn_network_config(
                    activation=self.activation,
                    out_activation=self.out_activation,
                    layer_width=self.layer_width,
                    num_layers=self.num_layers,
                ),
            )

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the MLP with hash encoding."""
        encoder = HashEncoding(
            num_levels=self.num_levels,
            min_res=self.min_res,
            max_res=self.max_res,
            log2_hashmap_size=self.log2_hashmap_size,
            features_per_level=self.features_per_level,
            hash_init_scale=self.hash_init_scale,
            implementation="torch",
        )
        mlp = MLP(
            in_dim=encoder.get_out_dim(),
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation="torch",
        )
        self.model = torch.nn.Sequential(encoder, mlp)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        return self.model(in_tensor)


class MLPWithMixedHashEncoding(FieldComponent):
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
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = 9

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

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("MLPWithHashEncoding")
            self.build_nn_modules()
        elif implementation == "tcnn":
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.in_dim,
                n_output_dims=self.out_dim,
                encoding_config=MixedHashEncoding.get_tcnn_encoding_config(
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
                ),
                network_config=MLP.get_tcnn_network_config(
                    activation=self.activation,
                    out_activation=self.out_activation,
                    layer_width=self.layer_width,
                    num_layers=self.num_layers,
                ),
            )

    def build_nn_modules(self) -> None:
        encoder = MixedHashEncoding(
            num_levels=self.num_levels,
            min_res=self.min_res,
            max_res=self.max_res,
            log2_hashmap_size=self.log2_hashmap_size,
            features_per_level=self.features_per_level,
            num_levels_2d=self.num_levels_2d,
            min_res_2d=self.min_res_2d,
            max_res_2d=self.max_res_2d,
            log2_hashmap_size_2d=self.log2_hashmap_size_2d,
            features_per_level_2d=self.features_per_level_2d,
            hash_init_scale=self.hash_init_scale,
            implementation="torch",
        )
        mlp = MLP(
            in_dim=encoder.get_out_dim(),
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation="torch",
        )
        self.model = torch.nn.Sequential(encoder, mlp)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        in_tensor = MixedHashEncoding.reorganize_input(in_tensor)
        return self.model(in_tensor)


class MLPWithSHEncoding(FieldComponent):
    def __init__(
        self,
        levels: int = 3,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        super().__init__()
        self.in_dim = 3

        self.levels = levels

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("MLPWithHashEncoding")
            self.build_nn_modules()
        elif implementation == "tcnn":
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.in_dim,
                n_output_dims=self.out_dim,
                encoding_config=SHEncoding.get_tcnn_encoding_config(levels=self.levels),
                network_config=MLP.get_tcnn_network_config(
                    activation=self.activation,
                    out_activation=self.out_activation,
                    layer_width=self.layer_width,
                    num_layers=self.num_layers,
                ),
            )

    def build_nn_modules(self) -> None:
        encoder = SHEncoding(levels=self.levels, implementation="torch")
        mlp = MLP(
            in_dim=encoder.get_out_dim(),
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation="torch",
        )
        self.model = torch.nn.Sequential(encoder, mlp)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        return self.model(in_tensor)


class MLPWithSHEncodingIdentity(FieldComponent):
    """
    The first 3 dimensions of the input are used for spherical harmonics encoding.
    """

    def __init__(
        self,
        in_dim: int,
        levels: int = 3,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        assert in_dim > 3, "identity_dim must be greater than 0"
        super().__init__()
        self.in_dim = in_dim
        self.identity_dim = in_dim - 3

        self.levels = levels

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            # print_tcnn_speed_warning("MLPWithHashEncoding")
            self.build_nn_modules()
        elif implementation == "tcnn":
            self.tcnn_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.in_dim,
                n_output_dims=self.out_dim,
                encoding_config=self.get_tcnn_encoding_config(levels=self.levels),
                network_config=MLP.get_tcnn_network_config(
                    activation=self.activation,
                    out_activation=self.out_activation,
                    layer_width=self.layer_width,
                    num_layers=self.num_layers,
                ),
            )

    def build_nn_modules(self) -> None:
        sh_encoder = SHEncoding(levels=self.levels, implementation="torch")
        mlp = MLP(
            in_dim=sh_encoder.get_out_dim() + self.identity_dim,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation="torch",
        )
        self.model = nn.ModuleDict({"sh_encoder": sh_encoder, "mlp": mlp})

    @classmethod
    def get_tcnn_encoding_config(cls, levels: int) -> dict:
        return {
            "otype": "Composite",
            "nested": [
                {"n_dims_to_encode": 3, **SHEncoding.get_tcnn_encoding_config(levels=levels)},
                {"otype": "Identity"},
            ],
        }

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        sh_encoding = self.model["sh_encoder"](in_tensor[..., :3])
        identity = in_tensor[..., 3:]
        x = torch.cat([sh_encoding, identity], dim=-1)
        return self.model["mlp"](x)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
