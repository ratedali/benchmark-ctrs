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

    @property
    def is_cifarresnet(self) -> bool:
        return "cifar_resnet" in self.value

    @property
    def is_resnet(self) -> bool:
        return "resnet" in self.value

    @property
    def resnet_depth(self) -> int:
        if not self.is_resnet:
            raise ValueError(
                "Architecture.resnet_depth is only supported on ResNet architectures"
            )
        return {
            "cifar_resnet18": 18,
            "resnet50": 50,
            "cifar_resnet110": 110,
        }[self.value]


__all__ = [
    "Architecture",
    "ArchitectureValues",
]
