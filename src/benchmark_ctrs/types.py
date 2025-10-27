from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from lightning.pytorch.utilities.types import (
        LRSchedulerConfig,
        OptimizerConfig,
        OptimizerLRSchedulerConfig,
    )
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
    from typing_extensions import NotRequired, TypeAlias

CONFIGURE_OPTIMIZERS: TypeAlias = """
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
"""

OptimizerCallable: TypeAlias = "Callable[[Iterable[Any]], Optimizer]"
LRSchedulerCallable: TypeAlias = "Callable[[Optimizer], LRScheduler|ReduceLROnPlateau]"
Criterion: TypeAlias = "Callable[[Tensor, Tensor], Tensor]"
CriterionCallable: TypeAlias = "Callable[[], Criterion]"

Batch: TypeAlias = "tuple[Tensor, ...]"


class StepOutput(TypedDict):
    loss: NotRequired[Tensor]
    predictions: NotRequired[Tensor]
