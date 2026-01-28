from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from torch.profiler import record_function
from typing_extensions import override

from benchmark_ctrs.modules.module import BaseModule
from benchmark_ctrs.types import Batch, StepOutput


class GaussianAug(BaseModule):
    def __init__(
        self,
        *args: Any,
        criterion: CrossEntropyLoss | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.criterion = criterion or CrossEntropyLoss()

    @override
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self.gaussian_aug_step(batch)

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self.gaussian_aug_step(batch)

    def gaussian_aug_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch[:2]
        with record_function("sampling"):
            sigma = self.hparams["sigma"]
            noisy_inputs = inputs + torch.randn_like(inputs) * sigma
            noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)
            predictions = self.forward(noisy_inputs)
        with record_function("classification_loss"):
            loss = self.criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
