"""Pooling Windows is a python library for generating foveated pooling windows."""

__all__ = [
    "PoolingWindows",
    "create_pooling_windows",
    "calculate",
    "pooling",
    "sampling",
]

from . import calculate, pooling, sampling
from .pooling import create_pooling_windows
from .pooling_windows import PoolingWindows


def __dir__() -> list[str]:
    return __all__
