from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, cast

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
from benchmark_ctrs.utilities import check_valid_step_output

if TYPE_CHECKING:
    from typing import Any

    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from benchmark_ctrs.types import CONFIGURE_OPTIMIZERS, Batch, StepOutput


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float
    lr_decay: float
    lr_step: int
    momentum: float
    weight_decay: float


class BaseModule(LightningModule, ABC):
    def __init__(
        self,
        *,
        arch: ArchitectureValues,
        num_classes: int,
        sds: list[float],
        means: list[float],
        params: HParams,
        cert: cr.Params | None = None,
        is_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(params))
        self.strict_loading = False

        self._num_classes = num_classes
        self._cert_params = cert

        self.__is_imagenet = is_imagenet
        self.__arch = Architecture.from_str(arch, source="value")
        self.__means = means
        self.__sds = sds

        self.automatic_accuracy: bool = True
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
        if (
            stage in {"fit", "validate"}
            and self._cert_params is not None
            and self.hparams["sigma"] > 0
        ):
            self._val_cert = FeatureShare(
                {
                    "certified_radius/average": cr.CertifiedRadius(
                        self._base_classifier,
                        self._cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="mean",
                    ),
                    "certified_radius/best": cr.CertifiedRadius(
                        self._base_classifier,
                        self._cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="max",
                    ),
                    "certified_radius/worst": cr.CertifiedRadius(
                        self._base_classifier,
                        self._cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="min",
                    ),
                }
            )

        self._predict_cert = None
        if stage == "predict" and self._cert_params and self.hparams["sigma"] > 0:
            self._predict_cert = cr.CertifiedRadius(
                self._base_classifier,
                self._cert_params,
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
        self, inputs: Tensor, *args: Any, add_noise: bool = True, **kwargs: Any
    ) -> Tensor:
        if add_noise:
            noises = torch.randn_like(inputs) * self.hparams["sigma"]
            return self._base_classifier(inputs + noises)
        return self._base_classifier(inputs)

    @override
    def on_train_start(self) -> None:
        super().on_train_start()
        if self.logger:
            keys = [
                f"{key}_{suffix}"
                for key in ("time/sec", "train/loss")
                for suffix in ("epoch", "step")
            ]
            metrics = dict.fromkeys(
                chain(
                    keys,
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
        if not check_valid_step_output(outputs):
            raise ValueError(
                "return value from `training_step` must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        self._batch_time(time.perf_counter() - self._batch_start)
        self.log("time/sec", self._batch_time, on_epoch=True)

        with torch.no_grad():
            if loss := outputs.get("loss"):
                self._loss_train(loss)

            if self.automatic_accuracy and (predictions := outputs.get("predictions")):
                self._acc_train.update(predictions, batch[1])

        if self._loss_train.update_called:
            self.log("train/loss", self._loss_train, on_epoch=True)

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if any(m.update_called for m in self._acc_train.values()):
            self.log_dict(self._acc_train)

    @override
    @abstractmethod
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput: ...

    @override
    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Batch, *args: Any, **kwargs: Any
    ) -> None:
        if not check_valid_step_output(outputs):
            raise ValueError(
                "return value from `validation_step` must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        inputs, targets = batch

        if loss := outputs.get("loss"):
            self._loss_val.update(loss)

        if self.automatic_accuracy and (predictions := outputs.get("predictions")):
            self._acc_val.update(predictions, targets)

        if self._val_cert is not None:
            self._val_cert.update(inputs, targets)

    @override
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self._loss_val.update_called:
            self.log("val/loss", self._loss_val, prog_bar=True)
        if any(m.update_called for m in self._acc_val.values()):
            self.log_dict(self._acc_val)
        if self._val_cert is not None and any(
            m.update_called for m in self._val_cert.values()
        ):
            self.log_dict(self._val_cert)

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def predict_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> cr.CertificationResult | None:
        if self._predict_cert:
            inputs, targets = batch
            self._predict_cert.update(inputs, targets)
            result = cast("cr.CertificationResult", self._predict_cert.compute())
            self._predict_cert.reset()
            return result
        return None

    def _default_eval_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs, add_noise=True)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
