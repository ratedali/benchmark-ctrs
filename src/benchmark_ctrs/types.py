from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
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


ConfigureOptimizers: TypeAlias = (
    Optimizer
    | Sequence[Optimizer]
    | tuple[
        Sequence[Optimizer],
        Sequence[LRScheduler | ReduceLROnPlateau | LRSchedulerConfig],
    ]
    | OptimizerConfig
    | OptimizerLRSchedulerConfig
    | Sequence[OptimizerConfig]
    | Sequence[OptimizerLRSchedulerConfig]
    | None
)


Classifier: TypeAlias = Callable[[Tensor], Tensor]
OptimizerCallable: TypeAlias = Callable[[Iterable[Any]], Optimizer]
LRSchedulerCallable: TypeAlias = Callable[[Optimizer], LRScheduler | ReduceLROnPlateau]
Batch: TypeAlias = tuple[Tensor, ...]
StepOutput: TypeAlias = DictStepOutput | None
