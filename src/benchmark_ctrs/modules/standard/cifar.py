from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from typing_extensions import override

from benchmark_ctrs.modules.standard.module import Standard
from benchmark_ctrs.types import LRScheduler
from benchmark_ctrs.utilities import GradualStepLR

__all__ = ["CIFAR_RESNET_DEPTH_WARMUP", "CIFARStandard"]

CIFAR_RESNET_DEPTH_WARMUP = 110


class CIFARStandard(Standard):
    @override
    def default_optimizer(self) -> Optimizer:
        return SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

    def default_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        if (
            self.model_architecture is not None
            and self.model_architecture.is_resnet
            and self.model_architecture.resnet_depth >= CIFAR_RESNET_DEPTH_WARMUP
        ):
            return GradualStepLR(
                optimizer,
                warmup_factor=0.1,
                warmup_iters=1,
                step_size=30,
                gamma=0.1,
            )
        return StepLR(optimizer, 30, 0.1)
