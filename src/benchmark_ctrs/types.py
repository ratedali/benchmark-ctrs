from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, Union

from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

if TYPE_CHECKING:
    from typing_extensions import NotRequired, TypeAlias

CONFIGURE_OPTIMIZERS: TypeAlias = Union[
    Optimizer,
    Sequence[Optimizer],
    tuple[
        Sequence[Optimizer],
        Sequence[Union[LRScheduler, ReduceLROnPlateau, LRSchedulerConfig]],
    ],
    OptimizerConfig,
    OptimizerLRSchedulerConfig,
    Sequence[OptimizerConfig],
    Sequence[OptimizerLRSchedulerConfig],
    None,
]

Batch: TypeAlias = tuple[Tensor, ...]


class StepOutput(TypedDict):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]
