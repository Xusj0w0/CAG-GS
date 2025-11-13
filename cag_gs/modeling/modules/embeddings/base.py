import sys
from abc import abstractmethod
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn


class _LazyError:
    def __init__(self, data):
        self.__data = data  # pylint: disable=unused-private-member

    class LazyErrorObj:
        def __init__(self, data):
            self.__data = data  # pylint: disable=unused-private-member

        def __call__(self, *args, **kwds):
            name, exc = object.__getattribute__(self, "__data")
            raise RuntimeError(f"Could not load package {name}.") from exc

        def __getattr__(self, __name: str):
            name, exc = object.__getattribute__(self, "__data")
            raise RuntimeError(f"Could not load package {name}") from exc

    def __getattr__(self, __name: str):
        return _LazyError.LazyErrorObj(object.__getattribute__(self, "__data"))


TCNN_EXISTS = False
tcnn_import_exception = None
tcnn = None
try:
    import tinycudann

    tcnn = tinycudann
    del tinycudann
    TCNN_EXISTS = True
except ModuleNotFoundError as _exp:
    tcnn_import_exception = _exp
except ImportError as _exp:
    tcnn_import_exception = _exp
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)
    tcnn_import_exception = _exp

if tcnn_import_exception is not None:
    tcnn = _LazyError(tcnn_import_exception)


class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError
