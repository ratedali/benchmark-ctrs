# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
#
# GitHub Permalink: https://github.com/locuslab/smoothing/blob/master/code/train.py

import torch
from typing_extensions import override

from benchmark_ctrs.training.methods.abc import (
    BatchResults,
    TrainingMethod,
)
from benchmark_ctrs.training.parameters import TrainingParameters


class StandardTraining(TrainingMethod[TrainingParameters]):
    """Implements standard Gaussian training from the paper:
    "Certified Adversarial Robustness via Randomized Smoothing", Cohen et al., 2019
    """

    @staticmethod
    @override
    def create_instance(params):
        return StandardTraining()

    @property
    @override
    def instance_tag(self):
        return ("standard",)

    def train(self, ctx, batch):
        # augment inputs with noise
        inputs = (
            batch.inputs
            + torch.randn_like(batch.inputs, device=ctx.device) * ctx.noise_sd
        )

        # compute predictions and loss
        predictions: torch.Tensor = ctx.model_wrapper.model(inputs)
        loss: torch.Tensor = ctx.criterion(predictions, batch.targets)

        # compute gradient and do SGD step
        ctx.optimizer.zero_grad()
        loss.mean().backward()
        ctx.optimizer.step()

        return BatchResults(predictions=predictions, loss=loss)
