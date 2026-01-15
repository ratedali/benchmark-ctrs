import dataclasses
import time
from itertools import chain
from typing import Any, Literal, Optional, Union, cast

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
from benchmark_ctrs.models import Architecture, ArchitectureValues
from benchmark_ctrs.models.cifar_resnet import resnet as cifar_resnet
from benchmark_ctrs.models.layers import Normalization
from benchmark_ctrs.models.lenet import LeNet
from benchmark_ctrs.types import Batch, ConfigureOptimizers, StepOutput
from benchmark_ctrs.utilities import check_valid_step_output


class PredictionResult(TypedDict, closed=True):
    certification: NotRequired[cr.CertificationResult]
    clean: Tensor


@dataclasses.dataclass(frozen=True)
class HParams:
    sigma: float


class BaseModule(L.LightningModule):
    __slots__ = ()

    automatic_accuracy: bool

    raw_model: nn.Module
    normalization_layer: Normalization
    model: nn.Module
    criterion: nn.Module

    model_architecture: Architecture
    num_classes: int
    grads_log_interval: int

    metric_batch_time: Metric
    metric_epoch_time: Metric
    metric_acc_train: MetricCollection
    metric_acc_val: MetricCollection
    metric_loss_train: Metric
    metric_loss_val: Metric
    metric_val_cert: Optional[cr.CertifiedRadius]
    metric_predict_cert: Optional[cr.CertifiedRadius]

    def __init__(
        self,
        *,
        num_classes: int,
        std: list[float],
        mean: list[float],
        params: HParams,
        certification: Union[CertificationMethod, Literal[True], None] = None,
        certification_params: Optional[cr.Params] = None,
        arch: Optional[ArchitectureValues] = None,
        default_arch: ArchitectureValues = "resnet50",
        criterion: Optional[nn.Module] = None,
        grads_log_interval: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(dataclasses.asdict(params))
        self.strict_loading = False
        self.automatic_accuracy = True

        self.num_classes = num_classes
        self.grads_log_interval = grads_log_interval

        self.model_architecture = cast(
            "Architecture",
            Architecture.from_str(
                arch or default_arch,
                source="value",
            ),
        )
        self.criterion = criterion or nn.CrossEntropyLoss(reduction="mean")

        self.init_model(list(mean), list(std))
        self.init_metrics(
            certification=certification,
            certification_params=certification_params,
        )

    def init_model(self, mean: list[float], std: list[float]) -> None:
        if self.model_architecture == Architecture.LeNet:
            self.raw_model = LeNet()
        elif self.model_architecture == Architecture.ResNet50:
            self.raw_model = models.resnet50(
                weights=None,
                num_classes=self.num_classes,
            )
        elif self.model_architecture.is_cifarresnet:
            self.raw_model = cifar_resnet(
                depth=self.model_architecture.resnet_depth, num_classes=self.num_classes
            )
        else:
            raise ValueError(
                f"Unknown value for arch: {self.model_architecture}. "
                f"Possible values are: {', '.join(Architecture._member_names_)}"
            )

        self.normalization_layer = Normalization(mean=mean, sd=std)
        self.model = torch.nn.Sequential(self.normalization_layer, self.raw_model)

    def init_metrics(
        self,
        certification: Union[cr.CertificationMethod, Literal[True], None],
        certification_params: Optional[cr.Params],
    ) -> None:
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

        self._batch_cuda_events: Optional[tuple[torch.cuda.Event, ...]] = None
        self.metric_batch_time = MeanMetric(nan_strategy="error")
        self.metric_epoch_time = SumMetric(nan_strategy="error")
        self.metric_gpu_memory = MeanMetric(nan_strategy="error")

        self.metric_acc_val = self.metric_acc_train.clone(prefix="val/")
        self.metric_loss_val = MeanMetric(nan_strategy="error")

        # Setup certified radius metrics
        sigma = float(self.hparams["sigma"])
        method: Optional[CertificationMethod] = (
            RSCertification() if certification is True else certification
        )

        self.metric_val_cert = None
        self.metric_predict_cert = None
        params = certification_params or cr.Params(sigma)
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

    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        optimizer = SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

        lr_scheduler = StepLR(optimizer, 50, 0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @property
    def eval_model(self) -> nn.Module:
        return self.model

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
    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> Optional[int]:
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

            self.log("certified_radius/average", cert.radii.mean())
            self.log("certified_radius/best", cert.radii.max())
            self.log("certified_radius/worst", cert.radii.min())

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
            result["certification"] = cert
        return result

    @rank_zero_only
    def log_grad_norms(self, norm_type: Union[float, str] = 2) -> None:
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
