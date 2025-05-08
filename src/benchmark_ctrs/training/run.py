import logging
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Generic

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import TypeVar

from benchmark_ctrs.dataset import DatasetWrapper
from benchmark_ctrs.model import ModelWrapper
from benchmark_ctrs.training.checkpoint import TrainingCheckpoint
from benchmark_ctrs.training.methods.abc import (
    Batch,
    TestingContext,
    TrainingContext,
    TrainingMethod,
)
from benchmark_ctrs.training.metrics import Metrics, ScalarTags
from benchmark_ctrs.training.parameters import TrainingParameters
from benchmark_ctrs.utils import AverageMetric, correct_pred

_logger = logging.getLogger(__name__)


_Tparams = TypeVar("_Tparams", bound=TrainingParameters)


class TrainingRun(Generic[_Tparams]):
    """
    Base class for training methods.
    """

    def __init__(
        self,
        id_: int,
        method: TrainingMethod[_Tparams],
        params: _Tparams,
    ):
        """Initialize the training run.
        This class is used to run the training loop and save checkpoints.

        Args:
            id_ (int): the id of the training run
            method (TrainingMethod[_Tparams]): the training method to use
            params (_Tparams): the training parameters
        """
        self.id = id_

        self._params = params
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._method = method
        self._dataset_wrapper = DatasetWrapper(params["dataset"], params["data_dir"])

        # example run directory:
        # "./runs/mnist/lenet/noise_1.00/adre_adv/k_8/lambda_2.0/"
        self._save_dir = (
            params["rundir"]
            .joinpath(
                params["dataset"],
                params["architecture"],
                f"noise_{params['noise_sd']:.2f}",
                *self._method.instance_tag,
                f"run_{params['id']}",
            )
            .resolve()
        )
        self._save = params["save"]
        if self._save or self._params["profiling"]:
            self._save_dir.mkdir(parents=True, exist_ok=True)

        self._last_epoch = -1
        checkpoint = None
        model_wrapper = None
        if params["resume"]:
            resume_path = params["resume_path"] or self._save_dir
            if resume_path.exists():
                checkpoint = TrainingCheckpoint.load(resume_path)
                self._last_epoch = checkpoint.epoch
                model_wrapper = ModelWrapper.get_wrapper(
                    checkpoint.arch,
                    self._dataset_wrapper,
                )
                model_wrapper.model.load_state_dict(checkpoint.state_dict)
            else:
                _logger.warning(
                    "Cannot resume training from: %s (checkpoint file doesn't exist)",
                    resume_path,
                )

        if model_wrapper is None:
            model_wrapper = ModelWrapper.get_wrapper(
                params["architecture"],
                self._dataset_wrapper,
            )
        self._model_wrapper = model_wrapper.to(self._device)

        if params["loss"] == "cross-entropy":
            self._loss = torch.nn.CrossEntropyLoss(reduction="none")
            _logger.debug("Using cross entropy loss")
        self._loss = self._loss.to(self._device)

        if params["optimizer"] == "sgd":
            self._optimizer = SGD(
                self._model_wrapper.model.parameters(),
                lr=params["lr"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )
            _logger.debug(
                "Using stochastic gradient descent "
                "(lr = %(lr).4g, "
                "momentum = %(momentum).4g, "
                "weight decay = %(weight_decay).4g)",
                params,
            )
        if checkpoint:
            self._optimizer.load_state_dict(checkpoint.optimizer)

        if params["lr_schedule"] == "constant":
            self._scheduler = None
            _logger.debug("using constant learning rate (%(lr).4g)", params)
        elif params["lr_schedule"] == "step":
            self._scheduler = StepLR(
                self._optimizer,
                step_size=params["lr_step_size"],
                gamma=params["lr_schedule_gamma"],
                last_epoch=self._last_epoch,
            )
            _logger.debug(
                "learining rate (initial = %(lr).4g) "
                "decays by %(lr_schedule_gamma).4g"
                "every %(lr_step_size)d epochs",
                params,
            )

    def run(self):
        """
        Run the training loop.
        """
        with self._setup_logging():
            pin_memory = self._params["dataset"] == "imagenet"

            val_pct = self._params["validation_set_split"]
            train_subset, val_subset = torch.utils.data.random_split(
                self._dataset_wrapper.get_split("train"),
                (1 - val_pct, val_pct),
            )
            train_loader = DataLoader(
                dataset=train_subset,
                shuffle=True,
                batch_size=self._params["batch_size"],
                num_workers=self._params["num_workers"],
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                dataset=val_subset,
                shuffle=True,
                batch_size=self._params["batch_size"],
                num_workers=self._params["num_workers"],
                pin_memory=pin_memory,
            )

            start_epoch = self._last_epoch + 1
            if start_epoch > 0:
                if start_epoch < self._params["epochs"]:
                    _logger.info("Resuming training from epoch %d", start_epoch + 1)
                else:
                    _logger.info("Training run has previously completed succesfully")
                    return

            with self._tensorboard() as writer:
                start_time = time.perf_counter()
                for epoch in range(start_epoch, self._params["epochs"]):
                    _logger.debug(
                        "Executing epoch training loop (%d/%d)",
                        epoch + 1,
                        self._params["epochs"],
                    )
                    before = time.perf_counter()
                    train_metrics = self.train(epoch, train_loader)
                    epoch_time = time.perf_counter() - before

                    _logger.debug(
                        "Calculating validation metrics after epoch (%d/%d)",
                        epoch + 1,
                        self._params["epochs"],
                    )
                    test_metrics = self.test(val_loader)

                    if self._scheduler is not None:
                        self._scheduler.step()

                    # Write the training checkpoint to disk
                    if self._save:
                        TrainingCheckpoint.capture(
                            self._model_wrapper,
                            self._optimizer,
                            epoch,
                        ).save(self._save_dir)

                    # Write the training and test metrics to disk
                    if writer is not None:
                        writer.add_scalar(ScalarTags.Time.Epoch, epoch_time, epoch)
                        for tag, value in train_metrics.scalars.items():
                            writer.add_scalar(f"{tag}/train", value, epoch)
                        for tag, value in test_metrics.scalars.items():
                            writer.add_scalar(f"{tag}/test", value, epoch)

                    lr = (
                        self._scheduler.get_last_lr()[0]
                        if self._scheduler is not None
                        else self._params["lr"]
                    )

                    _logger.info(
                        "[%(epoch)d/%(total)d]\t"
                        "Time %(epoch_time).3gs\t"
                        "Per Batch %(batch_time).3gs\t"
                        "Loss %(loss).4g\t"
                        "Acc@1 %(train_top1).2f/%(test_top1).2f\t"
                        "Acc@5 %(train_top5).2f/%(test_top5).2f\t"
                        "Learning Rate %(lr).4g",
                        {
                            "epoch": epoch + 1,
                            "total": self._params["epochs"],
                            "epoch_time": epoch_time,
                            "batch_time": train_metrics.batch_time,
                            "lr": lr,
                            "loss": train_metrics.loss,
                            "train_top1": train_metrics.top1_acc,
                            "test_top1": test_metrics.top1_acc,
                            "train_top5": train_metrics.top5_acc,
                            "test_top5": test_metrics.top5_acc,
                        },
                    )
            _logger.info("Done in %.3gs", time.perf_counter() - start_time)

    def train(
        self,
        epoch: int,
        loader: DataLoader[tuple[torch.Tensor, ...]],
    ):
        avg_data_time = AverageMetric()
        avg_batch_time = AverageMetric()
        avg_loss = AverageMetric()
        top1_acc = AverageMetric()
        top5_acc = AverageMetric()
        grads: dict[str, AverageMetric] = {}
        avg_extra: dict[str, AverageMetric] = {}

        ctx = TrainingContext(
            model_wrapper=self._model_wrapper,
            criterion=self._loss,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            epoch=epoch,
            noise_sd=self._params["noise_sd"],
            device=self._device,
        )

        ctx.model_wrapper.model.train()

        num_batches = len(loader)
        log_every = num_batches // self._params["log_freq"]

        with (
            self._profiling() as profiler,
            record_function("batch_training"),
        ):
            last_batch_end = time.perf_counter()
            for i, data in enumerate(loader):
                batch = Batch(data[0].to(ctx.device), data[1].to(ctx.device))
                avg_data_time.update(time.perf_counter() - last_batch_end)

                before = time.perf_counter()
                results = self._method.train(ctx, batch)
                avg_batch_time.update(time.perf_counter() - before)

                with torch.inference_mode():
                    avg_loss.add(results.loss)

                    correct_top1, correct_top5 = correct_pred(
                        results.predictions,
                        batch.targets,
                        ks=(1, 5),
                    )
                    top1_acc.add(correct_top1)
                    top5_acc.add(correct_top5)

                    if results.extra_metrics is not None:
                        batch_size = batch.inputs.size(0)
                        for key, val in results.extra_metrics.get_scalars().items():
                            if key not in avg_extra:
                                avg_extra[key] = AverageMetric()
                            avg_extra[key].update(val, batch_size)

                    if self._params["log_grads"]:
                        for (
                            name,
                            layer,
                        ) in self._model_wrapper.base_model.named_children():
                            if name not in grads:
                                grads[name] = AverageMetric()
                            layer_norm = torch.nn.utils.get_total_norm(
                                p.grad.detach()
                                for p in layer.parameters()
                                if p.grad is not None
                            )
                            grads[name].update(layer_norm.item())

                    if (i + 1) % log_every == 0:
                        _logger.info(
                            "%(progress).1f%%\t"
                            "Time %(batch_time).3gs\t"
                            "Loss %(loss).4g\t"
                            "Acc@1 %(train_top1).2f\t"
                            "Acc@5 %(train_top5).2f",
                            {
                                "progress": (i + 1) * 100 / num_batches,
                                "batch_time": avg_batch_time.value,
                                "loss": avg_loss.value,
                                "train_top1": top1_acc.value * 100,
                                "train_top5": top5_acc.value * 100,
                            },
                        )
                last_batch_end = time.perf_counter()
                if profiler is not None:
                    profiler.step()

        return Metrics(
            data_time=avg_data_time.value,
            batch_time=avg_batch_time.value,
            loss=avg_loss.value,
            top1_acc=top1_acc.value * 100,
            top5_acc=top5_acc.value * 100,
            extra={tag: acc.value for tag, acc in avg_extra.items()},
            layer_gradients={name: grad.value for name, grad in grads.items()},
        )

    @torch.inference_mode()
    def test(self, loader: DataLoader[tuple[torch.Tensor, ...]]):
        avg_loss = AverageMetric()
        top1_acc = AverageMetric()
        top5_acc = AverageMetric()
        avg_extra: dict[str, AverageMetric] = {}

        ctx = TestingContext(
            model_wrapper=self._model_wrapper,
            criterion=self._loss,
            noise_sd=self._params["noise_sd"],
            device=self._device,
        )

        ctx.model_wrapper.model.eval()

        for data in loader:
            batch = Batch(data[0].to(ctx.device), data[1].to(ctx.device))

            (pred, loss, extra) = self._method.test(ctx, batch)
            avg_loss.add(loss)

            correct_top1, correct_top2 = correct_pred(pred, batch.targets, ks=(1, 5))
            top1_acc.add(correct_top1)
            top5_acc.add(correct_top2)

            if extra is not None:
                batch_size = batch.inputs.size(0)
                for key, val in extra.get_scalars().items():
                    if key not in avg_extra:
                        avg_extra[key] = AverageMetric()
                    avg_extra[key].update(val, batch_size)

        return Metrics(
            loss=avg_loss.value,
            top1_acc=top1_acc.value * 100,
            top5_acc=top5_acc.value * 100,
            extra={tag: acc.value for tag, acc in avg_extra.items()},
        )

    @contextmanager
    def _profiling(self):
        ctxmgr = nullcontext()
        if self._params["profiling"]:

            def handler(p: profile):
                output = p.key_averages(
                    group_by_input_shape=True, group_by_stack_n=5
                ).table(sort_by=f"{self._device.type}_time_total", row_limit=50)
                with (self._save_dir / "profiling.log").open("ta") as logfile:
                    logfile.write(f"Torch profiling step {p.step_num}:" + os.linesep)
                    logfile.write(output)
                p.export_chrome_trace(str(self._save_dir / f"trace_{p.step_num}.json"))

            activities = [ProfilerActivity.CPU]
            if self._device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            ctxmgr = profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                schedule=schedule(skip_first=1, wait=10, warmup=10, active=5),
                on_trace_ready=handler,
            )
        with ctxmgr as p:
            yield p

    @contextmanager
    def _setup_logging(self):
        handler = None
        if self._save:
            path = self._save_dir / "run.log"
            _logger.info("Checkpoints, metrics and logs will be saved in: %s", path)

            handler = logging.FileHandler(path)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            _logger.addHandler(handler)
        try:
            yield _logger
        finally:
            if handler is not None:
                _logger.removeHandler(handler)
                handler.close()

    def _tensorboard(self):
        if self._save:
            return SummaryWriter(
                log_dir=str(self._save_dir),
            )
        return nullcontext()
