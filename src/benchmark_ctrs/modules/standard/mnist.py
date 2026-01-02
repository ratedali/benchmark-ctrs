from typing import Any
from torch.optim import SGD
from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule
from benchmark_ctrs.types import Batch, ConfigureOptimizers, StepOutput


class MNISTStandard(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            params=BaseHParams(sigma=0.0),
            **kwargs,
        )

    @override
    def configure_optimizers(self) -> ConfigureOptimizers:
        return SGD(self.parameters(), lr=self.hparams["learning_rate"])

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
