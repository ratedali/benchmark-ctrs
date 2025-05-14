from enum import Enum

from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.models.resnet import ResNet


class Architectures(Enum):
    LeNet = "lenet"
    Resnet_50 = "resnet50"
    Resnet_110 = "resnet110"


__all__ = ["LeNet", "ResNet"]
