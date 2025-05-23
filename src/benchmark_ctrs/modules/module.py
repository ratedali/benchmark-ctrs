from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypedDict

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
from benchmark_ctrs.models import Architectures
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


class BaseRandomizedSmoothing(L.LightningModule, ABC):
    def __init__(
        self,
        *,
        arch: Architectures,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        cert_val: certified_radius.Params | None = None,
        cert_test: certified_radius.Params | None = None,
        cert_pred: certified_radius.Params | None = None,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()

        self._num_classes = num_classes
        self.save_hyperparameters(dataclasses.asdict(params))
        self._val_cert_params = cert_val or certified_radius.Params(
            n0=10, n=500, max_=10
        )
        self._test_cert_params = cert_test or certified_radius.Params()
        self._pred_cert_params = cert_pred or certified_radius.Params()

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
        if self.__arch == Architectures.LeNet:
            self.__base_model = LeNet()
        elif self.__arch == Architectures.Resnet50:
            if self.__is_imagenet:
                self.__base_model = resnet50()
            else:
                self.__base_model = ResNet(depth=50, num_classes=self._num_classes)
        elif self.__arch == Architectures.Resnet110:
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
        if stage in {"fit", "validate"}:
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

    def _log_batch_metrics(
        self, outputs: STEP_OUTPUT, batch: Batch, *, stage: Literal["train", "validate"]
    ) -> None:
        if not is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        inputs, targets = batch
        loss, predictions = outputs["loss"], outputs["predictions"]
        if stage == "train":
            self._train_loss(loss)
            self.log("train/loss", self._train_loss, on_epoch=True, on_step=False)
            self._train_metrics(predictions, targets)
            self.log_dict(self._train_metrics, on_epoch=True, on_step=False)
        elif stage == "validate":
            self._val_loss(loss)
            self.log(
                "val/loss",
                self._train_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )
            self._val_metrics(predictions, targets)
            self.log_dict(self._val_metrics, on_epoch=True, on_step=False)
            self._val_cert(inputs)
            self.log_dict(self._val_cert, on_epoch=True, on_step=False)

    @override
    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    def predict_step(self, batch: Batch, *args: Any, **kwargs: Any) -> Any:
        return None

    @override
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._epoch_start = time.perf_counter()

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log(
            "time/epoch",
            time.perf_counter() - self._epoch_start,
            on_epoch=True,
            on_step=False,
        )

    @override
    def on_train_batch_start(self, batch: Any, *args: Any, **kwargs: Any) -> int | None:
        self._batch_start = time.perf_counter()

    @override
    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        self._log_batch_metrics(outputs, batch, stage="train")
        self._batch_time(time.perf_counter() - self._batch_start)
        self.log("time/batch", self._batch_time, on_epoch=False, on_step=True)

    @override
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        self._log_batch_metrics(outputs, batch, stage="validate")
