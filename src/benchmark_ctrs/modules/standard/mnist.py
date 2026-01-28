from torch.optim import SGD
from typing_extensions import override

from benchmark_ctrs.modules.standard.module import Standard
from benchmark_ctrs.types import ConfigureOptimizers

__all__ = ["MNISTStandard"]


class MNISTStandard(Standard):
    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        return SGD(self.parameters(), lr=0.1)
