import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from typing_extensions import override

from benchmark_ctrs.modules.rs_training import RSTrainingModule


class GaussianAug(RSTrainingModule):
    @override
    def training_step(self, batch: tuple[Tensor, ...]) -> STEP_OUTPUT:
        inputs, targets = batch
        inputs = inputs + torch.randn_like(inputs) * self.hparams["sigma"]

        # compute predictions and loss
        predictions = self.forward(inputs)
        loss = self._criterion(predictions, targets)
        return {
            "prediction": predictions,
            "loss": loss,
        }
