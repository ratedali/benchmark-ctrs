from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Union

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from typing_extensions import NotRequired, TypedDict

ConfigureOptimizers = Union[
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

Batch = tuple[Tensor, ...]


class DictStepOutput(TypedDict, extra_items=Any):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]


StepOutput = Optional[DictStepOutput]
