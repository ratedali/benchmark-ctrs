from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    import torch

_logger = logging.getLogger(__name__)


def correct_pred(pred: torch.Tensor, target: torch.Tensor, ks: tuple[int, ...] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    topk = pred.topk(max(ks))
    is_topk_correct = topk.indices == target.view(-1, 1)

    res: list[torch.Tensor] = []
    for k in ks:
        is_k_correct = is_topk_correct[:, :k].sum(dim=1)
        res.append(is_k_correct)
    return res


class AverageMetric:
    def __init__(self):
        self._sum: float = 0.0
        self._count: int = 0

    @property
    def sum(self):
        return self._sum

    @property
    def count(self):
        return self._count

    @property
    def value(self):
        if self._count > 0:
            return self._sum / self._count
        return 0.0

    def add(self, value: torch.Tensor) -> Self:
        self._sum += value.sum().item()
        self._count += value.size(0)
        return self

    def update(self, avg: float, count: int = 1) -> Self:
        self._sum += avg * count
        self._count += count
        return self
