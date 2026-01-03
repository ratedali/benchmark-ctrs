from typing import Any

import torch
from torch import Tensor
from torch.profiler import record_function
from typing_extensions import override

from benchmark_ctrs.modules import BaseModule
from benchmark_ctrs.types import Batch, StepOutput


class GaussianAug(BaseModule):
    @override
    def forward(
        self,
        inputs: Tensor,
        *args: Any,
        add_noise: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        sigma = self.hparams["sigma"]
        if add_noise and sigma != 0:
            noises = torch.randn_like(inputs) * sigma
            inputs = torch.clamp(inputs + noises, 0.0, 1.0)
        return super().forward(inputs, *args, **kwargs)

    @override
    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self.gaussian_aug_step(batch)

    @override
    def validation_step(self, batch: Batch, *args: Any, **kwargs: Any) -> StepOutput:
        return self.gaussian_aug_step(batch)

    def gaussian_aug_step(self, batch: Batch) -> StepOutput:
        inputs, targets = batch[:2]
        with record_function("sampling"):
            predictions = self.forward(inputs, add_noise=True)
        with record_function("classification_loss"):
            loss = self.criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
