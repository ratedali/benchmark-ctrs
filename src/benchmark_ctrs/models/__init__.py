from typing import Literal

from lightning.pytorch.utilities import LightningEnum

ArchitectureValues = Literal[
    "lenet",
    "cifar_resnet18",
    "cifar_resnet110",
    "resnet50",
]


class Architecture(LightningEnum):
    LeNet = "lenet"
    CIFARResNet18 = "cifar_resnet18"
    CIFARResNet110 = "cifar_resnet110"
    ResNet50 = "resnet50"


__all__ = [
    "Architecture",
    "ArchitectureValues",
]
