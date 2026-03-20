import math

import torch
import torch.nn as nn

from .base import tcnn
from .encodings import HashEncoding, MixedHashEncoding, SHEncoding
from .mlps import (MLP, MixedHashEncoding, MLPWithHashEncoding,
                   MLPWithMixedHashEncoding, MLPWithSHEncoding,
                   MLPWithSHEncodingIdentity)


def _init_tcnn_encoding(module: nn.Module, params: torch.Tensor):
    if isinstance(module, (HashEncoding, MixedHashEncoding, MLPWithHashEncoding, MLPWithMixedHashEncoding)):
        scale = getattr(module, "hash_init_scale", 0.001)
        nn.init.uniform_(params.data, -scale, scale)
    elif isinstance(module, (SHEncoding, MLPWithSHEncoding, MLPWithSHEncodingIdentity)):
        nn.init.constant_(params.data, 0.0)
    else:
        raise NotImplementedError("Unsupported tcnn encoding")


def _init_tcnn_network(module: nn.Module, params: torch.Tensor):
    layer_width = getattr(module, "layer_width", 64)
    # Kaiming Uniform Bound: sqrt(6 / fan_in)
    bound = math.sqrt(6.0 / layer_width)
    nn.init.uniform_(params.data, -bound, bound)


def initialize_weights(module: nn.Module):
    # For tcnn implementation
    if hasattr(module, "tcnn_model") and module.tcnn_model is not None:
        tcnn_obj = module.tcnn_model
        # [Encoding | Network]
        if isinstance(tcnn_obj, tcnn.NetworkWithInputEncoding):
            n_enc_params = module.n_encoding_params
            if n_enc_params > 0:
                _init_tcnn_encoding(module, tcnn_obj.params[:n_enc_params])
            if tcnn_obj.params.shape[0] > n_enc_params:
                _init_tcnn_network(module, tcnn_obj.params[n_enc_params:])
        elif isinstance(tcnn_obj, tcnn.Network):
            _init_tcnn_network(module, tcnn_obj.params)
        elif isinstance(tcnn_obj, tcnn.Encoding):
            _init_tcnn_encoding(module, tcnn_obj.params)
        return

    # For torch implementation
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, HashEncoding):
        if hasattr(module, "hash_table") and isinstance(module.hash_table, nn.Parameter):
            scale = getattr(module, "hash_init_scale", 1e-3)
            nn.init.uniform_(module.hash_table.data, -scale, scale)

    # recursively init children
    for child in module.children():
        initialize_weights(child)
