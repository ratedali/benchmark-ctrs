from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing_extensions import override

from benchmark_ctrs.modules.standard.module import Standard
from benchmark_ctrs.types import ConfigureOptimizers

__all__ = ["ImageNetStandard"]


class ImageNetStandard(Standard):
    @override
    def init_model(self, mean: list[float], std: list[float]) -> None:
        super().init_model(mean, std)
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        optimizer = SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.min_epochs or 90,
            eta_min=0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
