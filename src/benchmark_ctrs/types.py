from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.optim import Optimizer, lr_scheduler
from typing_extensions import NotRequired, TypedDict

__all__ = [
    "Batch",
    "Classifier",
    "ConfigureOptimizers",
    "DictStepOutput",
    "LRSchedulerCallable",
    "OptimizerCallable",
    "StepOutput",
]


class DictStepOutput(TypedDict, extra_items=Any):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]


Classifier: TypeAlias = Callable[[Tensor], Tensor]
OptimizerCallable: TypeAlias = Callable[[Iterable[Any]], Optimizer]
LRScheduler: TypeAlias = lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau
LRSchedulerCallable: TypeAlias = Callable[[Optimizer], LRScheduler]
Batch: TypeAlias = tuple[Tensor, ...]
StepOutput: TypeAlias = DictStepOutput | None


ConfigureOptimizers: TypeAlias = (
    Optimizer
    | Sequence[Optimizer]
    | tuple[
        Sequence[Optimizer],
        Sequence[LRScheduler | LRSchedulerConfig],
    ]
    | OptimizerConfig
    | OptimizerLRSchedulerConfig
    | Sequence[OptimizerConfig]
    | Sequence[OptimizerLRSchedulerConfig]
    | None
)
