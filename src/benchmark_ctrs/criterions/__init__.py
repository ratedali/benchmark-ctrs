from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from torch import Tensor

__all__ = ["Criterion", "CriterionCallable", "Reduction"]

Reduction = Literal["mean", "sum", "none"]


class Criterion(Protocol):
    def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor: ...


T_co = TypeVar("T_co", covariant=True, bound=Criterion)


class CriterionCallable(Protocol, Generic[T_co]):
    def __call__(self, reduction: str) -> T_co: ...


AnyCriterionCallable: TypeAlias = CriterionCallable[Criterion]
