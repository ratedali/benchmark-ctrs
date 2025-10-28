from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypedDict, Union

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from typing_extensions import NotRequired, TypeAlias

CONFIGURE_OPTIMIZERS: TypeAlias = Union[
    Optimizer,
    Sequence[Optimizer],
    tuple[
        Sequence[Optimizer],
        Sequence[
            Union[
                LRScheduler,
                ReduceLROnPlateau,
                LRSchedulerConfig,
            ]
        ],
    ],
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
    Sequence[OptimizerConfig],
    Sequence[OptimizerLRSchedulerConfig],
    None,
]

Classifier: TypeAlias = Callable[[Tensor], Tensor]
OptimizerCallable: TypeAlias = Callable[[Iterable[Any]], Optimizer]
LRSchedulerCallable: TypeAlias = Callable[
    [Optimizer], Union[LRScheduler, ReduceLROnPlateau]
]
Criterion: TypeAlias = Callable[[Tensor, Tensor], Tensor]
CriterionCallable: TypeAlias = Callable[[], Criterion]

Batch: TypeAlias = tuple[Tensor, ...]


class StepOutput(TypedDict):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]
