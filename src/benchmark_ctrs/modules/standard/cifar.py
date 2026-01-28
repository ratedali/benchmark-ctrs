from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from typing_extensions import override

from benchmark_ctrs.modules.standard.module import Standard
from benchmark_ctrs.types import ConfigureOptimizers
from benchmark_ctrs.utilities import GradualStepLR

__all__ = ["CIFAR_RESNET_DEPTH_WARMUP", "CIFARStandard"]

CIFAR_RESNET_DEPTH_WARMUP = 110


class CIFARStandard(Standard):
    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        optimizer = SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

        if (
            self.model_architecture.is_resnet
            and self.model_architecture.resnet_depth >= CIFAR_RESNET_DEPTH_WARMUP
        ):
            lr_scheduler = GradualStepLR(
                optimizer,
                warmup_factor=0.1,
                warmup_iters=1,
                step_size=30,
                gamma=0.1,
            )
        else:
            lr_scheduler = StepLR(optimizer, 30, 0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
