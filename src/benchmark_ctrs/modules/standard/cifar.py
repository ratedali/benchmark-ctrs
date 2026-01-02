from typing import Any

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule
from benchmark_ctrs.types import Batch, ConfigureOptimizers, StepOutput
from benchmark_ctrs.utilities import GradualStepLR

WARMUP_DEPTH_THRESHOLD = 110


class CIFARStandard(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, params=BaseHParams(sigma=0), **kwargs)

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
            and self.model_architecture.resnet_depth >= WARMUP_DEPTH_THRESHOLD
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

    @override
    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "predictions": predictions,
        }

    @override
    def validation_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "predictions": predictions,
        }
