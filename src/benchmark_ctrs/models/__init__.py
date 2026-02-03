# ruff: noqa: F403
import re
from typing import Literal

from benchmark_ctrs.models import layers, lenet, resnet
from benchmark_ctrs.models.layers import *
from benchmark_ctrs.models.lenet import *
from benchmark_ctrs.models.resnet import *

__all__ = ["resnet_arch_info"]
__all__ += resnet.__all__
__all__ += layers.__all__
__all__ += lenet.__all__


_resnet_depth_pat = re.compile(r"(?P<is_cifar>cifar-)?resnet(?P<depth>\d+)")


def make_resnet_arch(dataset: Literal["imagenet", "cifar"], depth: int):
    if dataset == "cifar":
        return f"cifar-resnet{depth}"
    return f"resnet{depth}"


def resnet_arch_info(arch: str) -> tuple[Literal["cifar", "imagenet"], int] | None:
    if match := _resnet_depth_pat.match(arch):
        is_cifar, depth = match.group("is_cifar", "depth")
        return "cifar" if is_cifar else "imagenet", int(depth)
    return None
