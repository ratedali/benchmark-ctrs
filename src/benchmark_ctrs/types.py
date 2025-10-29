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
from typing_extensions import NotRequired

CONFIGURE_OPTIMIZERS = Union[
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

Classifier = Callable[[Tensor], Tensor]
OptimizerCallable = Callable[[Iterable[Any]], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], Union[LRScheduler, ReduceLROnPlateau]]
Criterion = Callable[[Tensor, Tensor], Tensor]
CriterionCallable = Callable[[], Criterion]

Batch = tuple[Tensor, ...]


class StepOutput(TypedDict):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]
