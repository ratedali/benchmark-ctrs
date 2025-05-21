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
from torchmetrics import Metric, MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy
from torchmetrics.wrappers import FeatureShare
from torchvision.models import resnet50
from typing_extensions import override

from benchmark_ctrs.metrics import CertifiedRadius
from benchmark_ctrs.models import (
    Architectures,
    LeNet,
    ResNet,
    smooth,
)
from benchmark_ctrs.models.layers import Normalization

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Union

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
    Batch: TypeAlias = tuple[Tensor, ...]


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float
    lr_decay: float
    lr_step: int
    momentum: float
    weight_decay: float


class StepOutput(TypedDict):
    loss: Tensor
    predictions: Tensor


def is_valid_step_output(value: Any) -> TypeIs[StepOutput]:
    return (
        isinstance(value, dict)
        and isinstance(value.get("loss"), Tensor)
        and isinstance(value.get("predictions"), Tensor)
    )


class RandomizedSmoothing(L.LightningModule, ABC):
    def __init__(
        self,
        *,
        arch: Architectures,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        cert_val: smooth.HParams | None = None,
        cert_test: smooth.HParams | None = None,
        cert_pred: smooth.HParams | None = None,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()

        self._num_classes = num_classes
        self.save_hyperparameters(dataclasses.asdict(params))
        self._val_cert_params = cert_val or smooth.HParams(n0=10, n=500)
        self._test_cert_params = cert_test or smooth.HParams()
        self._pred_cert_params = cert_pred or smooth.HParams()

        self.__is_imagenet = is_imagenet
        self.__arch = arch
        self.__means = means
        self.__sds = sds

        self._train_metrics = MetricCollection(
            {"accuracy": Accuracy(task="multiclass", num_classes=self._num_classes)},
            prefix="train/",
        )
        self._train_loss = MeanMetric()
        self._batch_time = MeanMetric()

        self._val_metrics = self._train_metrics.clone(prefix="val/")
        self._val_loss = MeanMetric()

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

        self._criterion = torch.nn.CrossEntropyLoss()
        self.__norm_layer = Normalization(mean=self.__means, sd=self.__sds)
        self._base_classifier = torch.nn.Sequential(
            self.__norm_layer, self.__base_model
        )
        self._val_cert = FeatureShare(
            {
                "acr": CertifiedRadius(
                    self._base_classifier,
                    self._val_cert_params,
                    num_classes=self._num_classes,
                    sigma=self.hparams["sigma"],
                    reduction="mean",
                ),
                "best_cr": CertifiedRadius(
                    self._base_classifier,
                    self._val_cert_params,
                    num_classes=self._num_classes,
                    sigma=self.hparams["sigma"],
                    reduction="max",
                ),
                "worst_cr": CertifiedRadius(
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
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + torch.randn_like(inputs) * self.hparams["sigma"]
        return self._base_classifier(inputs)

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

    def _default_eval_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}

    def _log_batch_metrics(
        self,
        outputs: STEP_OUTPUT,
        batch: Batch,
        *,
        prefix: str | None,
        loss_metric: Metric,
        acc_metrics: MetricCollection,
    ):
        if not is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        loss_metric(outputs["loss"])
        self.log(
            f"{prefix or ''}loss",
            loss_metric,
            prog_bar=True,
            on_epoch=True,
        )

        _inputs, targets = batch
        acc_metrics(outputs["predictions"], targets)
        self.log_dict(acc_metrics, on_epoch=True)

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
        predictions = self._base_classifier(batch)
        return torch.argmax(predictions, dim=1)

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
            loss_metric=self._train_loss,
            acc_metrics=self._train_metrics,
            prefix=self._train_metrics.prefix,
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
            loss_metric=self._val_loss,
            acc_metrics=self._val_metrics,
            prefix=self._val_metrics.prefix,
        )
        self._val_cert(batch[0])
        self.log_dict(self._val_cert, on_epoch=True)
