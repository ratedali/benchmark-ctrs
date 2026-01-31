import os
import time
from itertools import chain
from typing import Any, Literal, cast

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm, rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection, SumMetric
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy
from torchvision import models
from typing_extensions import NotRequired, TypedDict, override

from benchmark_ctrs.certification import CertificationMethod
from benchmark_ctrs.certification.rs_certification import RSCertification
from benchmark_ctrs.metrics import certified_radius as cr
from benchmark_ctrs.models import Architecture, ArchitectureOption
from benchmark_ctrs.models.layers import Normalization
from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.models.resnet import cifar_resnet
from benchmark_ctrs.types import (
    Batch,
    ConfigureOptimizers,
    LRScheduler,
    LRSchedulerCallable,
    OptimizerCallable,
    StepOutput,
)
from benchmark_ctrs.utilities import check_valid_step_output

__all__ = [
    "BaseModule",
    "PredictionResult",
]


class PredictionResult(TypedDict, closed=True):
    certification: NotRequired[cr.CertificationResult]
    clean: Tensor


class BaseModule(L.LightningModule):
    __slots__ = ()

    automatic_accuracy: bool

    raw_model: nn.Module
    normalization_layer: Normalization
    model: nn.Module

    model_architecture: Architecture | None
    num_classes: int
    grads_log_interval: int

    metric_batch_time: Metric
    metric_epoch_time: Metric
    metric_acc_train: MetricCollection
    metric_acc_val: MetricCollection
    metric_loss_train: Metric
    metric_loss_val: Metric
    metric_val_cert: cr.CertifiedRadius | None
    metric_predict_cert: cr.CertifiedRadius | None

    def __init__(
        self,
        sigma: float = 0,
        *,
        num_classes: int,
        std: list[float],
        mean: list[float],
        certification: CertificationMethod | Literal[True] | None = None,
        certification_params: cr.Params | None = None,
        arch: ArchitectureOption | str | None = None,
        default_arch: ArchitectureOption | str = "resnet50",
        grads_log_interval: int = 0,
        optimizer: OptimizerCallable | None,
        lr_scheduler: LRSchedulerCallable | None,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.strict_loading = False
        self.automatic_accuracy = True

        # Fields
        self.num_classes = num_classes
        self.grads_log_interval = grads_log_interval

        self.architecture = arch or default_arch
        self.norm_mean = list(mean)
        self.norm_std = list(std)
        self.certification_method: CertificationMethod | None | Literal[True] = (
            certification
        )
        self.certification_params = certification_params
        self.optimizer_callable = optimizer
        self.lr_scheduler_callable = lr_scheduler

        # Metrics
        self.metric_acc_train = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
                "accuracy_top5": Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    top_k=5,
                ),
            },
            prefix="train/",
        )
        self.metric_loss_train = MeanMetric(nan_strategy="error")

        self.metric_acc_val = self.metric_acc_train.clone(prefix="val/")
        self.metric_loss_val = MeanMetric(nan_strategy="error")

        self._batch_cuda_events: tuple[torch.cuda.Event, ...] | None = None
        self.metric_batch_time = MeanMetric(nan_strategy="error")
        self.metric_epoch_time = SumMetric(nan_strategy="error")
        self.metric_gpu_memory = MeanMetric(nan_strategy="error")

    @override
    def setup(self, stage: str) -> None:
        super().setup(stage)

        # set max number of cpus
        cpus = os.environ.get("NUM_AVAILABLE_CPUS", None)
        if cpus is not None:
            local_world_size = int(
                os.environ.get(
                    "LOCAL_WORLD_SIZE",
                    self.trainer.world_size // self.trainer.num_nodes,
                )
            )

            # use manual number of cpus to impose restrictions
            max_cpus = max(1, int(cpus) // local_world_size - 1)
            torch.set_num_threads(max_cpus)
            torch.set_num_interop_threads(max_cpus)

    @override
    def configure_model(self) -> None:
        if not hasattr(self, "model"):
            self.init_model()
            self.init_cert()

    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        if self.optimizer_callable:
            optimizer = self.optimizer_callable(self.parameters())
        else:
            optimizer = self.default_optimizer()

        if self.lr_scheduler_callable:
            lr_scheduler = self.lr_scheduler_callable(optimizer)
        else:
            lr_scheduler = self.default_lr_scheduler(optimizer)

        if lr_scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    @property
    def eval_model(self) -> nn.Module:
        return self.model

    def init_model(self) -> None:
        self.model_architecture = cast(
            "Architecture | None",
            Architecture.try_from_str(self.architecture, source="value"),
        )
        if self.model_architecture is not None:
            if self.model_architecture == Architecture.LeNet:
                self.raw_model = LeNet()
            elif self.model_architecture == Architecture.ResNet50:
                self.raw_model = models.resnet50(
                    weights=None,
                    num_classes=self.num_classes,
                )
            elif self.model_architecture.is_cifarresnet:
                self.raw_model = cifar_resnet(
                    depth=self.model_architecture.resnet_depth,
                    num_classes=self.num_classes,
                )
        if not self.raw_model:
            try:
                self.raw_model = models.get_model(self.architecture)
            except KeyError as e:
                raise ValueError(
                    f"Unknown value for arch: {self.architecture}. "
                    f"Possible values are: {', '.join(Architecture._member_names_)} "
                    "or a valid torchvision model name."
                ) from e

        self.normalization_layer = Normalization(mean=self.norm_mean, sd=self.norm_std)
        self.model = torch.nn.Sequential(self.normalization_layer, self.raw_model)

    def init_cert(self) -> None:
        # Setup certified radius metrics
        method: CertificationMethod | None = (
            RSCertification()
            if self.certification_method is True
            else self.certification_method
        )

        self.metric_val_cert = None
        self.metric_predict_cert = None
        params = self.certification_params or cr.Params(self.sigma)
        if method is not None:
            self.metric_val_cert = cr.CertifiedRadius(
                self.eval_model,
                method,
                params,
                num_classes=self.num_classes,
                reduction="none",
            )

        if method is not None:
            self.metric_predict_cert = cr.CertifiedRadius(
                self.eval_model,
                method,
                params,
                num_classes=self.num_classes,
                reduction="none",
            )

    def default_optimizer(self) -> Optimizer:
        return SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    def default_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        return StepLR(optimizer, 50, 0.1)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        if self.training:
            return self.model(*args, **kwargs)

        return self.eval_model(*args, **kwargs)

    @override
    def on_train_start(self) -> None:
        super().on_train_start()
        if self.logger:
            keys = [
                f"{key}_{suffix}"
                for key in ("time/iteration", "train/loss")
                for suffix in ("epoch", "step")
            ]
            keys.extend(["time/epoch", "train/gpu_memory"])
            metrics = dict.fromkeys(
                chain(
                    keys,
                    self.metric_acc_train.keys(),
                    self.metric_acc_val.keys(),
                    [f"certified_radius/{cr}" for cr in ("average", "best", "worst")],
                ),
                0.0,
            )
            self.logger.log_hyperparams(dict(self.hparams), metrics)

    @override
    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> int | None:
        super().on_train_batch_start(batch, batch_idx)
        self._batch_start = time.perf_counter()
        if torch.cuda.is_available():
            start = cast("torch.cuda.Event", torch.cuda.Event(enable_timing=True))
            end = cast("torch.cuda.Event", torch.cuda.Event(enable_timing=True))
            self._batch_cuda_events = (start, end)
            start.record()

    @override
    def on_train_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if not check_valid_step_output(outputs):
            raise ValueError(
                "return value from `training_step` must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        if self._batch_cuda_events:
            start, end = self._batch_cuda_events
            end.record()
            torch.cuda.synchronize()
            iteration_time = start.elapsed_time(end) / 1000.0

            self.metric_gpu_memory(torch.cuda.memory_allocated())
            self.log("train/gpu_memory", self.metric_gpu_memory)
        else:
            iteration_time = time.perf_counter() - self._batch_start

        self.metric_batch_time(iteration_time)
        self.metric_epoch_time.update(iteration_time)
        self.log("time/iteration", self.metric_batch_time, on_epoch=True)

        if outputs is not None:
            with torch.no_grad():
                if "loss" in outputs:
                    loss = outputs["loss"]
                    if loss.dim() > 0:
                        batch_size = batch[0].size(0)
                        loss = loss.sum() / batch_size
                    self.metric_loss_train(loss.item())

                if self.automatic_accuracy and "predictions" in outputs:
                    self.metric_acc_train.update(outputs["predictions"], batch[1])

        if self.metric_loss_train.update_called:
            self.log("train/loss", self.metric_loss_train, on_epoch=True)

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        interval = self.grads_log_interval
        if interval > 0 and self.trainer.global_step % interval == 0:
            self.log_grad_norms()

    @override
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.log("time/epoch", self.metric_epoch_time)
        if any(m.update_called for m in self.metric_acc_train.values()):
            self.log_dict(self.metric_acc_train)

    @override
    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        super().training_step(batch, *args, **kwargs)

    @override
    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().on_validation_batch_end(outputs, batch, batch_idx, *args, **kwargs)
        if not check_valid_step_output(outputs):
            raise ValueError(
                "return value from `validation_step` must be a dict with the tensors "
                f"'loss' and 'predictions', got value: {outputs}"
            )

        inputs, targets, *_ = batch

        if outputs is not None:
            if "loss" in outputs:
                loss = outputs["loss"]
                if loss.dim() > 0:
                    batch_size = batch[0].size(0)
                    loss = loss.sum() / batch_size
                self.metric_loss_val.update(loss.item())

            if self.automatic_accuracy and "predictions" in outputs:
                self.metric_acc_val.update(outputs["predictions"], targets)

        if self.metric_val_cert is not None:
            self.metric_val_cert.update(inputs, targets)

    @override
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self.metric_loss_val.update_called:
            self.log("val/loss", self.metric_loss_val, prog_bar=True)

        if any(m.update_called for m in self.metric_acc_val.values()):
            self.log_dict(self.metric_acc_val)

        if self.metric_val_cert is not None and self.metric_val_cert.update_called:
            cert = cast("cr.CertificationResult", self.metric_val_cert.compute())
            self.metric_val_cert.reset()

            if self.trainer.is_global_zero:
                self.log("certified_radius/average", cert.radii.mean(), sync_dist=False)
                self.log("certified_radius/best", cert.radii.max(), sync_dist=False)
                self.log("certified_radius/worst", cert.radii.min(), sync_dist=False)

    @override
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput: ...

    @override
    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput: ...

    @override
    def predict_step(self, batch: Batch, *args: Any, **kwargs: Any) -> PredictionResult:
        inputs = batch[0]
        result: PredictionResult = {
            "clean": self.forward(inputs).argmax(dim=1),
        }
        if self.metric_predict_cert:
            self.metric_predict_cert.update(inputs)
            cert = cast("cr.CertificationResult", self.metric_predict_cert.compute())
            self.metric_predict_cert.reset()

            if self.trainer.is_global_zero:
                result["certification"] = cert
        return result

    @rank_zero_only
    def log_grad_norms(self, norm_type: float | str = 2) -> None:
        """Utility to compute and log grad norms.

        It gets called in the before_optimizer_step hook,
        if automatic optimization is used.

        Needs to be called manually, when using manual
        optimization.
        """
        total_key = f"grad_{norm_type}_norm_total"
        norms = grad_norm(self.model, norm_type=norm_type)
        self.log(
            "backprop/grad_l2_norm_total",
            norms.get(total_key, 0.0),
        )

        if isinstance(self.logger, TensorBoardLogger):
            tensorboard = cast("SummaryWriter", self.logger.experiment)
            tensorboard.add_histogram(
                tag="backprop/grad_l2_norms",
                values=torch.tensor([v for k, v in norms.items() if k != total_key]),
                global_step=self.trainer.global_step,
            )
