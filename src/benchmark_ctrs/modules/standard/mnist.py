from torch.optim import SGD, Optimizer
from typing_extensions import override

from benchmark_ctrs.modules.standard.module import Standard
from benchmark_ctrs.types import LRScheduler

__all__ = ["MNISTStandard"]


class MNISTStandard(Standard):
    @override
    def default_optimizer(self) -> Optimizer:
        return SGD(self.parameters(), lr=0.1)

    @override
    def default_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        return None
