from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy
from torchmetrics.wrappers import FeatureShare
from torchvision.models import resnet50
from typing_extensions import override

from benchmark_ctrs.metrics import certified_radius as cr
from benchmark_ctrs.models import Architecture, ArchitectureValues
from benchmark_ctrs.models.layers import Normalization
from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.models.resnet import CIFARResNet

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
    from typing_extensions import NotRequired, TypeAlias, TypeIs

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
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float
    lr_decay: float
    lr_step: int
    momentum: float
    weight_decay: float


class BaseRandomizedSmoothing(LightningModule, ABC):
    def __init__(
        self,
        *,
        arch: ArchitectureValues,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        cert_val: cr.Params | None = None,
        cert_predict: cr.Params | None = None,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(params))
        self.strict_loading = False

        self._num_classes = num_classes
        self.__val_cert_params = cert_val
        self.__predict_cert_params = cert_predict

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

    @override
    def setup(self, stage: str) -> None:
        if self.__arch == Architecture.LeNet:
            self.__base_model = LeNet()
        elif self.__arch == Architecture.Resnet50:
            if self.__is_imagenet:
                self.__base_model = resnet50()
            else:
                self.__base_model = CIFARResNet(depth=50, num_classes=self._num_classes)
        elif self.__arch == Architecture.Resnet110:
            self.__base_model = CIFARResNet(depth=110, num_classes=self._num_classes)
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
        if stage in {"fit", "validate"} and self.__val_cert_params is not None:
            self._val_cert = FeatureShare(
                {
                    "certified_radius/average": cr.CertifiedRadius(
                        self._base_classifier,
                        self.__val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="mean",
                    ),
                    "certified_radius/best": cr.CertifiedRadius(
                        self._base_classifier,
                        self.__val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="max",
                    ),
                    "certified_radius/worst": cr.CertifiedRadius(
                        self._base_classifier,
                        self.__val_cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="min",
                    ),
                }
            )
        if stage == "predict" and self.__predict_cert_params:
            self._predict_cert = cr.CertifiedRadius(
                self._base_classifier,
                self.__predict_cert_params,
                num_classes=self._num_classes,
                sigma=self.hparams["sigma"],
                reduction="none",
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
    def forward(
        self, inputs: Tensor, *args: Any, noise: bool = True, **kwargs: Any
    ) -> Tensor:
        if noise:
            noises = torch.randn_like(inputs) * self.hparams["sigma"]
            return self._base_classifier(inputs + noises)
        return self._base_classifier(inputs)

    @override
    def on_train_start(self) -> None:
        super().on_train_start()
        if self.logger:
            metrics = dict.fromkeys(
                chain(
                    ["time/sec", "train/loss", "val/loss"],
                    self._acc_train.keys(),
                    self._acc_val.keys(),
                    self._val_cert.keys() if self._val_cert else (),
                ),
                0.0,
            )
            self.logger.log_hyperparams(dict(self.hparams), metrics)

    @override
    def on_train_batch_start(self, *args: Any, **kwargs: Any) -> int | None:
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

        if "loss" in outputs:
            self._loss_train(outputs["loss"].detach())
        self.log("train/loss", self._loss_train, on_epoch=True)

        if "predictions" in outputs:
            _inputs, targets = batch
            self._acc_train.update(outputs["predictions"].detach(), targets)

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log_dict(self._acc_train)

    @override
    @abstractmethod
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput: ...

    @override
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        if not BaseRandomizedSmoothing.__is_valid_step_output(outputs):
            raise ValueError(
                "step output must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )
        if "loss" in outputs:
            self._loss_val.update(outputs["loss"])

        inputs, targets = batch
        if "predictions" in outputs:
            self._acc_val.update(outputs["predictions"], targets)

        if self._val_cert is not None:
            self._val_cert.update(inputs, targets)

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
            and ("loss" not in value or isinstance(value["loss"], Tensor))
            and isinstance(value["predictions"], Tensor)
        )

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def predict_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> cr.CertificationResult:
        inputs, targets = batch
        self._predict_cert.update(inputs, targets)
        result = cast("cr.CertificationResult", self._predict_cert.compute())
        self._predict_cert.reset()
        return result

    def _default_eval_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
