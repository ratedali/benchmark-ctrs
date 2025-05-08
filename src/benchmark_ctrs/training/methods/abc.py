from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

import torch
from typing_extensions import NamedTuple, TypeVar

from benchmark_ctrs.training.parameters import TrainingParameters

if TYPE_CHECKING:
    from benchmark_ctrs.model import ModelWrapper
    from benchmark_ctrs.training.metrics import SupportsScalars


class Batch(NamedTuple):
    inputs: torch.Tensor
    targets: torch.Tensor


class BatchResults(NamedTuple):
    predictions: torch.Tensor
    loss: torch.Tensor
    extra_metrics: SupportsScalars | None = None


class TrainingContext(NamedTuple):
    model_wrapper: ModelWrapper
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    epoch: int
    noise_sd: float
    device: torch.device


class TestingContext(NamedTuple):
    model_wrapper: ModelWrapper
    criterion: torch.nn.Module
    noise_sd: float
    device: torch.device


_Tparams = TypeVar("_Tparams", bound=TrainingParameters)


class TrainingMethod(ABC, Generic[_Tparams]):
    """Base class for training methods.
    This class is used to define the method-specific elements of the training loop
    """

    @staticmethod
    @abstractmethod
    def create_instance(
        params: _Tparams,
    ) -> TrainingMethod: ...

    @property
    @abstractmethod
    def instance_tag(self) -> tuple[str, ...]:
        """gives a unique ordered tag for the relevent subset of method parameters

        Returns:
            tuple[str, ...]: the tag
        """

    @abstractmethod
    def train(self, ctx: TrainingContext, batch: Batch) -> BatchResults:
        """
        Train the model using one batch cycle.

        Args:
            ctx (TrainingContext): contains all current training run details.
            batch (Batch): a batch sampled from the labeled training set

        Returns:
            BatchResults: The batch predictions, training loss, and metrics for batch.
        """

    def test(self, ctx: TestingContext, batch: Batch) -> BatchResults:
        """
        Test the model performance on one data batch

        Args:
            ctx (TestingContext): contains all current training run details.
            batch (Batch): a batch sampled from the labeled target test set

        Returns:
            BatchResults: The batch predictions, test loss, and metrics for batch.
        """
        noisy_inputs = (
            batch.inputs
            + torch.randn_like(batch.inputs, device=ctx.device) * ctx.noise_sd
        )

        predictions: torch.Tensor = ctx.model_wrapper.model(noisy_inputs)
        loss: torch.Tensor = ctx.criterion(predictions, batch.targets)

        return BatchResults(predictions, loss)
