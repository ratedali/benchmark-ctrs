from typing import Any

from torch.nn import CrossEntropyLoss
from typing_extensions import override

from benchmark_ctrs.modules.module import BaseModule
from benchmark_ctrs.types import Batch, StepOutput

__all__ = ["Standard"]


class Standard(BaseModule):
    def __init__(
        self,
        *args: Any,
        criterion: CrossEntropyLoss | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=self.ignore_hyperparameters)
        self.criterion = criterion or CrossEntropyLoss()

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
