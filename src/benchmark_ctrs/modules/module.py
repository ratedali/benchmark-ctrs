from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict

import lightning as L
import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy
from torchmetrics.wrappers import FeatureShare
from torchvision.models import resnet50
from typing_extensions import override

from benchmark_ctrs.metrics import certified_radius
from benchmark_ctrs.models import Architecture, ArchitectureValues
from benchmark_ctrs.models.layers import Normalization
from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.models.resnet import ResNet

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Union

    from lightning.pytorch.utilities.types import (
        STEP_OUTPUT,
        LRSchedulerConfig,
        OptimizerConfig,
        OptimizerLRSchedulerConfig,
    )
    from torch.nn import Module
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
    from typing_extensions import TypeAlias, TypeIs

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

Batch: TypeAlias = tuple[Tensor, ...]


class StepOutput(TypedDict):
    loss: Tensor
    predictions: Tensor


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float
    lr_decay: float
    lr_step: int
    momentum: float
    weight_decay: float


class BaseRandomizedSmoothing(L.LightningModule, ABC):
    def __init__(
        self,
        *,
        arch: ArchitectureValues,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        cert_val: certified_radius.Params | None = None,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(params))

        self._num_classes = num_classes
        self._val_cert_params = cert_val

        self.__is_imagenet = is_imagenet
        self.__arch = Architecture.from_str(arch, source="value")
        self.__means = means
        self.__sds = sds

        self._acc_train = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=self._num_classes),
            },
            prefix="train/",
        )
        self._loss_train = MeanMetric()
        self._batch_time = MeanMetric()

        self._acc_val = self._acc_train.clone(prefix="val/")
        self._loss_val = MeanMetric()

    @property
    def base_classifier(self) -> Module:
        return self._base_classifier

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def sigma(self) -> float:
        return self.hparams["sigma"]

    @override
    def setup(self, stage: str) -> None:
        if self.__arch == Architecture.LeNet:
            self.__base_model = LeNet()
        elif self.__arch == Architecture.Resnet50:
            if self.__is_imagenet:
                self.__base_model = resnet50()
            else:
                self.__base_model = ResNet(depth=50, num_classes=self._num_classes)
        elif self.__arch == Architecture.Resnet110:
            self.__base_model = ResNet(depth=110, num_classes=self._num_classes)
        else:
            raise ValueError(
                f"Unknown value for arch: {self.__arch}. "
                f"Possible values are: {', '.join(Architecture._member_names_)}"
            )

        self._criterion = torch.nn.CrossEntropyLoss()
        self.__norm_layer = Normalization(mean=self.__means, sd=self.__sds)
        self._base_classifier = torch.nn.Sequential(
            self.__norm_layer, self.__base_model
        )

        self._val_cert = None
        if stage in {"fit", "validate"} and self._val_cert_params is not None:
            self._val_cert = FeatureShare(
                {
                    "acr": certified_radius.CertifiedRadius(
                        self._base_classifier,
                        self._val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="mean",
                    ),
                    "best_cr": certified_radius.CertifiedRadius(
                        self._base_classifier,
                        self._val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="max",
                    ),
                    "worst_cr": certified_radius.CertifiedRadius(
                        self._base_classifier,
                        self._val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="min",
                    ),
                }
            )

    @override
    def configure_optimizers(self) -> CONFIGURE_OPTIMIZERS:
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams_initial.learning_rate,
            momentum=self.hparams_initial.momentum,
            weight_decay=self.hparams_initial.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams_initial.lr_step,
            gamma=self.hparams_initial.lr_decay,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @override
    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        inputs = inputs + torch.randn_like(inputs) * self.hparams["sigma"]
        return self._base_classifier(inputs)

    @override
    @abstractmethod
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput: ...

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    def _default_eval_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}

    @override
    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    def predict_step(self, batch: Batch, *args: Any, **kwargs: Any) -> Any:
        return None

    @override
    def on_train_batch_start(self, batch: Any, *args: Any, **kwargs: Any) -> int | None:
        self._batch_start = time.perf_counter()

    @override
    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        if not BaseRandomizedSmoothing.__is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        self._batch_time(time.perf_counter() - self._batch_start)
        self.log("time/sec", self._batch_time, on_epoch=True)

        self._loss_train(outputs["loss"].detach())
        self.log("train/loss", self._loss_train, on_epoch=True)

        _inputs, targets = batch
        self._acc_train.update(outputs["predictions"].detach(), targets)

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log_dict(self._acc_train)

    @override
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        if not BaseRandomizedSmoothing.__is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        self._loss_val.update(outputs["loss"])

        inputs, targets = batch
        self._acc_val.update(outputs["predictions"], targets)

        if self._val_cert is not None:
            self._val_cert.update(inputs)

    @override
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.log("val/loss", self._loss_val, prog_bar=True)
        self.log_dict(self._acc_val)
        if self._val_cert is not None:
            self.log_dict(self._val_cert)

    @staticmethod
    def __is_valid_step_output(value: Any) -> TypeIs[StepOutput]:
        return (
            isinstance(value, dict)
            and isinstance(value.get("loss"), Tensor)
            and isinstance(value.get("predictions"), Tensor)
        )
