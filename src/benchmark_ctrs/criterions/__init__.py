from typing import Literal, Protocol

from torch import Tensor

__all__ = ["Criterion", "CriterionCallable", "Reduction"]

Reduction = Literal["mean", "sum", "none"]


class Criterion(Protocol):
    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor: ...


class CriterionCallable(Protocol):
    def __call__(self, reduction: Reduction) -> Criterion: ...
