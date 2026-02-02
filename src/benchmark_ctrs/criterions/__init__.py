from typing import Literal, Protocol, TypeAlias

from torch import nn

__all__ = ["Criterion", "CriterionCallable", "Reduction"]

Reduction = Literal["mean", "sum", "none"]


Criterion: TypeAlias = nn.Module


class CriterionCallable(Protocol):
    def __call__(self, *, reduction: str) -> Criterion: ...
