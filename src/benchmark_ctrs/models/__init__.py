from typing import Literal

from lightning.pytorch.utilities import LightningEnum

from benchmark_ctrs.models import layers, lenet, resnet

__all__ = [
    "Architecture",
    "ArchitectureOption",
]
__all__ += resnet.__all__
__all__ += layers.__all__
__all__ += lenet.__all__

ArchitectureOption = Literal[
    "lenet",
    "cifar-resnet20",
    "cifar-resnet110",
    "resnet50",
]


class Architecture(LightningEnum):
    LeNet = "lenet"
    CIFARResNet20 = "cifar-resnet20"
    CIFARResNet110 = "cifar-resnet110"
    ResNet50 = "resnet50"

    @property
    def is_cifarresnet(self) -> bool:
        return "cifar-resnet" in self.value

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
            "cifar-resnet20": 20,
            "resnet50": 50,
            "cifar-resnet110": 110,
        }[self.value]
