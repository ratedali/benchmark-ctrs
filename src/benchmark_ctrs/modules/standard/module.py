from typing import Any

from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule
from benchmark_ctrs.types import Batch, StepOutput


class Standard(BaseModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, params=BaseHParams(sigma=0.0))

    @override
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "predictions": predictions,
        }

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs)
        loss = self.criterion(predictions, targets)
        return {
            "loss": loss,
            "predictions": predictions,
        }
