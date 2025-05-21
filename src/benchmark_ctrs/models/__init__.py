from lightning.pytorch.utilities import LightningEnum

from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.models.resnet import ResNet


class Architectures(LightningEnum):
    LeNet = "lenet"
    Resnet50 = "resnet50"
    Resnet110 = "resnet110"


__all__ = ["LeNet", "ResNet"]
