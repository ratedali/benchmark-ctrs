from typing import Literal

from lightning.pytorch.utilities import LightningEnum

ArchitectureValues = Literal["lenet", "resnet50", "resnet110"]


class Architecture(LightningEnum):
    LeNet = "lenet"
    Resnet50 = "resnet50"
    Resnet110 = "resnet110"


__all__ = [
    "Architecture",
    "ArchitectureValues",
]
