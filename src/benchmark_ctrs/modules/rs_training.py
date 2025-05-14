from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import lightning as L
import torch
from torch.optim import SGD
from torchvision.models import resnet50
from typing_extensions import override

from benchmark_ctrs.models import (
    Architectures,
    LeNet,
    Normalization,
    ResNet,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Union

    from lightning.pytorch.utilities.types import (
        STEP_OUTPUT,
        LRSchedulerConfig,
        OptimizerConfig,
        OptimizerLRSchedulerConfig,
    )
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
    from typing_extensions import TypeAlias

    CONFIGURE_OPTIMIZERS: TypeAlias = Union[
        torch.optim.Optimizer,
        Sequence[torch.optim.Optimizer],
        tuple[
            Sequence[torch.optim.Optimizer],
            Sequence[Union[LRScheduler, ReduceLROnPlateau, LRSchedulerConfig]],
        ],
        OptimizerConfig,
        OptimizerLRSchedulerConfig,
        Sequence[OptimizerConfig],
        Sequence[OptimizerLRSchedulerConfig],
        None,
    ]


@dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float = 0.1


class RSTrainingModule(L.LightningModule):
    def __init__(
        self,
        *,
        arch: Architectures,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()

        self._num_classes = num_classes
        self.save_hyperparameters(asdict(params))

        self.__is_imagenet = is_imagenet
        self.__arch = arch
        self.__means = means
        self.__sds = sds

    @override
    def setup(self, stage: str) -> None:
        if self.__arch == Architectures.LeNet:
            self.__base_model = LeNet()
        elif self.__arch == Architectures.Resnet_50:
            if self.__is_imagenet:
                self.__base_model = resnet50()
            else:
                self.__base_model = ResNet(depth=50, num_classes=self._num_classes)
        elif self.__arch == Architectures.Resnet_110:
            self.__base_model = ResNet(depth=110, num_classes=self._num_classes)
        else:
            raise ValueError(
                f"Unknown value for arch: {self.__arch}. "
                f"Possible values are: {', '.join(Architectures._member_names_)}"
            )

        self.__norm_layer = Normalization(mean=self.__means, sd=self.__sds)
        self._model = torch.nn.Sequential(self.__norm_layer, self.__base_model)
        self._criterion = torch.nn.CrossEntropyLoss()

    @override
    def configure_optimizers(self) -> CONFIGURE_OPTIMIZERS:
        return SGD(self.parameters(), self.hparams["learning_rate"])

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)

    @override
    def validation_step(self, batch: tuple[torch.Tensor, ...]) -> STEP_OUTPUT:
        inputs, targets = batch
        inputs = inputs + torch.randn_like(inputs) * self.hparams["sigma"]

        # compute predictions and loss
        predictions: torch.Tensor = self.forward(inputs)
        return self._criterion(predictions, targets)

    @override
    def test_step(self, batch: tuple[torch.Tensor, ...]) -> STEP_OUTPUT:
        inputs, targets = batch
        inputs = inputs + torch.randn_like(inputs).to(inputs) * self.hparams["sigma"]

        # compute predictions and loss
        predictions: torch.Tensor = self.forward(inputs)
        return self._criterion(predictions, targets)
