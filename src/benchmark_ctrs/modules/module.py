from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    ConstantLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.optim.optimizer import Optimizer
from torch.profiler import record_function
from torchmetrics import Metric, MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy
from torchmetrics.wrappers import FeatureShare
from torchvision import models
from typing_extensions import TypeAlias, override

from benchmark_ctrs.metrics import certified_radius as cr
from benchmark_ctrs.models import Architecture, ArchitectureValues
from benchmark_ctrs.models.cifar_resnet import ResNet
from benchmark_ctrs.models.cifar_resnet import resnet as cifar_resnet
from benchmark_ctrs.models.layers import Normalization
from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.utilities import check_valid_step_output

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from tensorboardX import SummaryWriter

    from benchmark_ctrs.types import CONFIGURE_OPTIMIZERS, Batch, StepOutput


WARMUP_DEPTH_THRESHOLD = 110


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float
    learning_rate: float
    lr_decay: float
    lr_step: int
    momentum: float
    weight_decay: float


OptimizerCallable: TypeAlias = Callable[[Iterable[Any]], Optimizer]
LRSchedulerCallable: TypeAlias = Callable[
    [Optimizer],
    Union[LRScheduler, ReduceLROnPlateau],
]


def warmup_lr_scheduler(
    optimizer: Optimizer, step_size: int, gamma: float, **kwargs
) -> LRScheduler:
    warmup_phase = ConstantLR(optimizer, factor=0.1, total_iters=1)
    normal_phase = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return SequentialLR(
        optimizer, [warmup_phase, normal_phase], milestones=[1], **kwargs
    )


class BaseModule(LightningModule, ABC):
    optimizer: OptimizerCallable
    lr_scheduler: LRSchedulerCallable | None
    automatic_accuracy: bool

    _arch: Architecture
    _cert_params: cr.Params | None
    _num_classes: int
    _mean: list[float]
    _std: list[float]
    _grads_log_interval: int
    _norm: Normalization
    _raw_model: nn.Module
    _model: nn.Module
    _criterion: nn.Module

    _batch_time: Metric
    _acc_train: MetricCollection
    _acc_val: MetricCollection
    _loss_train: Metric
    _loss_val: Metric
    _val_cert: MetricCollection | None
    _predict_cert: cr.CertifiedRadius | None

    def __init__(
        self,
        *,
        num_classes: int,
        std: list[float],
        mean: list[float],
        params: HParams,
        cert: cr.Params | None = None,
        arch: ArchitectureValues | None = None,
        default_arch: ArchitectureValues = "resnet50",
        optimizer: OptimizerCallable | None = None,
        lr_scheduler: LRSchedulerCallable | bool = True,
        grads_log_interval: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(params))
        self.strict_loading = False

        self._num_classes = num_classes
        self._cert_params = cert
        self._grads_log_interval = grads_log_interval

        self._arch = cast(
            "Architecture",
            Architecture.from_str(
                arch or default_arch,
                source="value",
            ),
        )
        self._mean = list(mean)
        self._std = list(std)

        self.optimizer = (
            optimizer
            if optimizer is not None
            else partial(
                SGD,
                lr=self.hparams["learning_rate"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"],
            )
        )

        default_lr_scheduler = partial(
            warmup_lr_scheduler
            if (
                self._arch.is_resnet
                and self._arch.resnet_depth >= WARMUP_DEPTH_THRESHOLD
            )
            else StepLR,
            step_size=self.hparams["lr_step"],
            gamma=self.hparams["lr_decay"],
        )
        self.lr_scheduler = (
            default_lr_scheduler
            if lr_scheduler is True
            else lr_scheduler
            if lr_scheduler is not False
            else None
        )

        self.automatic_accuracy = True
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
        if self._arch == Architecture.LeNet:
            self._raw_model = LeNet()
        elif self._arch == Architecture.ResNet50:
            self._raw_model = models.resnet50(
                weights=None,
                num_classes=self._num_classes,
            )
        elif self._arch.is_cifarresnet:
            self._raw_model = cifar_resnet(
                depth=self._arch.resnet_depth, num_classes=self._num_classes
            )
        else:
            raise ValueError(
                f"Unknown value for arch: {self._arch}. "
                f"Possible values are: {', '.join(Architecture._member_names_)}"
            )

        self._criterion = torch.nn.CrossEntropyLoss()
        self._norm = Normalization(mean=self._mean, sd=self._std)
        self._model = torch.nn.Sequential(self._norm, self._raw_model)

        self._val_cert = None
        if (
            stage in {"fit", "validate"}
            and self._cert_params is not None
            and self.hparams["sigma"] > 0
        ):
            self._val_cert = FeatureShare(
                {
                    "certified_radius/average": cr.CertifiedRadius(
                        self._model,
                        self._cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="mean",
                    ),
                    "certified_radius/best": cr.CertifiedRadius(
                        self._model,
                        self._cert_params,
                        num_classes=self._num_classes,
                        sigma=self.hparams["sigma"],
                        reduction="max",
                    ),
                    "certified_radius/worst": cr.CertifiedRadius(
                        self._model,
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
                self._model,
                self._cert_params,
                num_classes=self._num_classes,
                sigma=self.hparams["sigma"],
                reduction="none",
            )

    @override
    def configure_optimizers(self) -> CONFIGURE_OPTIMIZERS:
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler(optimizer),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    @override
    def forward(
        self, inputs: Tensor, *args: Any, add_noise: bool = True, **kwargs: Any
    ) -> Tensor:
        sigma = self.hparams["sigma"]
        if add_noise and sigma > 0:
            noises = torch.randn_like(inputs) * sigma
            return self._model(inputs + noises)
        return self._model(inputs)

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
            if (loss := outputs.get("loss")) is not None:
                if loss.dim() > 0:
                    batch_size = batch[0].size(0)
                    loss = loss.sum() / batch_size
                self._loss_train(loss.item())

            if (
                self.automatic_accuracy
                and (predictions := outputs.get("predictions")) is not None
            ):
                self._acc_train.update(predictions, batch[1])

        if self._loss_train.update_called:
            self.log("train/loss", self._loss_train, on_epoch=True)

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        interval = self._grads_log_interval
        if interval > 0 and self.trainer.global_step % interval == 0:
            self.log_grad_norms()

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

        inputs, targets, *_ = batch

        if (loss := outputs.get("loss")) is not None:
            if loss.dim() > 0:
                batch_size = batch[0].size(0)
                loss = loss.sum() / batch_size
            self._loss_val.update(loss.item())

        if (
            self.automatic_accuracy
            and (predictions := outputs.get("predictions")) is not None
        ):
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
        return self._default_eval_step(batch, add_noise=True)

    @override
    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self._default_eval_step(batch, add_noise=True)

    @override
    def predict_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> cr.CertificationResult | None:
        if self._predict_cert:
            inputs, targets, *_ = batch
            self._predict_cert.update(inputs, targets)
            result = cast("cr.CertificationResult", self._predict_cert.compute())
            self._predict_cert.reset()
            return result
        return None

    def _default_eval_step(
        self, batch: Batch, *, add_noise: bool = False
    ) -> StepOutput:
        inputs, targets, *_ = batch
        with record_function("sampling"):
            predictions = self.forward(inputs, add_noise=add_noise)
        with record_function("classification_loss"):
            loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}

    def log_grad_norms(self, norm_type: float | str = 2) -> None:
        """Utility to compute and log grad norms.

        It gets called in the before_optimizer_step hook,
        if automatic optimization is used.

        Needs to be called manually, when using manual
        optimization.

        """
        norms = grad_norm(self._model, norm_type=norm_type)
        self.log(
            "backprop/grad_l2_norm_total",
            norms.pop("grad_2.0_norm_total", 0.0),
        )

        if isinstance(self.logger, TensorBoardLogger):
            tensorboard = cast("SummaryWriter", self.logger.experiment)
            tensorboard.add_histogram(
                tag="backprop/grad_l2_norms",
                values=torch.tensor(list(norms.values())),
                global_step=self.trainer.global_step,
            )
