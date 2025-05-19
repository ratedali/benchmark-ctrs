from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import lightning as L
import torch
import torchmetrics
import torchmetrics.aggregation
import torchmetrics.classification
from torch.optim import SGD
from torchvision.models import resnet50
from typing_extensions import override

from benchmark_ctrs.models import (
    Architectures,
    LeNet,
    ResNet,
)
from benchmark_ctrs.models.layers import Normalization

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
    Batch: TypeAlias = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float = 0.1


class StepOutput(TypedDict):
    loss: torch.Tensor
    predictions: torch.Tensor


def is_valid_step_output(value: Any) -> TypeIs[StepOutput]:
    return (
        isinstance(value, dict)
        and isinstance(value.get("loss"), torch.Tensor)
        and isinstance(value.get("predictions"), torch.Tensor)
    )


class RSTrainingModule(L.LightningModule, ABC):
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

        self._train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=self._num_classes
                )
            },
            prefix="train_",
        )
        self._train_loss = torchmetrics.aggregation.MeanMetric()
        self._batch_time = torchmetrics.aggregation.MeanMetric()

        self._val_metrics = self._train_metrics.clone(prefix="val_")
        self._val_loss = torchmetrics.aggregation.MeanMetric()

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
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._epoch_start = time.perf_counter()

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("epoch_time", time.perf_counter() - self._epoch_start)

    @override
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        self._batch_start = time.perf_counter()
        return super().on_train_batch_start(batch, batch_idx)

    @override
    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, batch_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._log_batch_metrics(
            outputs,
            batch,
            prefix="train_",
            loss_metric=self._train_loss,
            acc_metrics=self._train_metrics,
        )
        self._batch_time(time.perf_counter() - self._batch_start)
        self.log("batch_time", self._batch_time, on_epoch=True)

    @override
    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
        self._log_batch_metrics(
            outputs,
            batch,
            prefix="val_",
            loss_metric=self._val_loss,
            acc_metrics=self._val_metrics,
        )

    def _log_batch_metrics(
        self,
        outputs: STEP_OUTPUT,
        batch: Batch,
        *,
        prefix: str,
        loss_metric: torchmetrics.Metric,
        acc_metrics: torchmetrics.MetricCollection,
    ):
        if not is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        loss_metric(outputs["loss"])
        self.log(
            f"{prefix}loss",
            loss_metric,
            prog_bar=True,
            on_epoch=True,
        )

        _inputs, targets = batch
        acc_metrics(outputs["predictions"], targets)
        self.log_dict(acc_metrics, prog_bar=True, on_epoch=True)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + torch.randn_like(inputs) * self.hparams["sigma"]
        return self._model(inputs)

    @override
    @abstractmethod
    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput: ...

    @override
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._default_eval_step(batch)

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ):
        predictions = self._model(batch)
        return torch.argmax(predictions, dim=1)

    def _default_eval_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
