from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.profiler import record_function
from torchmetrics.aggregation import MeanMetric
from typing_extensions import override

from benchmark_ctrs.modules.rs_training import HParams, RandomizedSmoothing

if TYPE_CHECKING:
    from torch import Tensor
    from typing_extensions import Literal

    from benchmark_ctrs.modules.rs_training import (
        Batch,
        StepOutput,
    )


@dataclasses.dataclass(frozen=True)
class MACERHParams(HParams):
    learning_rate: float = 0.01
    lr_decay: float = 0.1
    lr_step: int = 50
    momentum: float = 0.9
    weight_decay: float = 1e-4
    deferred: bool = True

    k: int = 16
    lbd: float = 12
    beta: float = 16
    gamma: float = 8


class MACER(RandomizedSmoothing):
    def __init__(
        self,
        *args,
        params: MACERHParams,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, params=params)
        self._norm = Normal(0.0, 1.0)
        self._cl = {
            "train": MeanMetric(),
            "val": MeanMetric(),
        }
        self._rl = {
            "train": MeanMetric(),
            "val": MeanMetric(),
        }

    @override
    def training_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        skip_rl = (
            self.hparams["deferred"] and self.current_epoch < self.hparams["lr_step"]
        )
        return self._loss(batch, skip_rl=skip_rl, prefix="train")

    @override
    def validation_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._loss(batch, prefix="val")

    def _loss(
        self,
        batch: Batch,
        *,
        skip_rl: bool = False,
        prefix: Literal["train", "val"] | None = None,
    ) -> StepOutput:
        hparams = cast("MACERHParams", self.hparams)
        inputs, targets = batch

        batch_size = inputs.size(0)
        new_shape = [batch_size * hparams.k]
        new_shape.extend(inputs[0].shape)

        with record_function("sampling"):
            samples = inputs.repeat((1, hparams.k, 1, 1)).view(new_shape)
            predictions = self.forward(samples).reshape((batch_size, hparams.k, -1))

        with record_function("classification_loss"):
            # perturbed loss
            pred_softmax = torch.softmax(predictions, dim=2).mean(1)
            pred_logsoftmax = torch.log(pred_softmax + 1e-10)  # avoid nan
            cl = F.nll_loss(pred_logsoftmax, targets, reduction="mean")

        with record_function("robust_loss"):
            rl = torch.zeros_like(targets)
            if not skip_rl and hparams.lbd != 0:
                # only apply beta to the robustness loss
                beta_pred = predictions * hparams.beta
                beta_pred_softmax = torch.softmax(beta_pred, dim=2).mean(1)

                top2 = torch.topk(beta_pred_softmax, 2)
                pA, pB = top2.values[:, 0], top2.values[:, 1]

                with torch.no_grad():
                    correct = top2.indices[:, 0] == targets  # G_theta

                    zeta_tmp = torch.zeros_like(rl)
                    zeta_tmp[correct] = self._norm.icdf(pA[correct]) - self._norm.icdf(
                        pB[correct],
                    )
                    # apply hinge and discard nan and inf values
                    nonzero = (
                        correct
                        & ~torch.isnan(zeta_tmp)
                        & ~torch.isinf(zeta_tmp)
                        & (torch.abs(zeta_tmp) <= hparams.gamma)
                    )
                zeta = self._norm.icdf(pA[nonzero]) - self._norm.icdf(pB[nonzero])
                rl[nonzero] = hparams.sigma * (hparams.gamma - zeta) / 2
            rl = rl.mean()

        if prefix is not None:
            self._cl[prefix](cl)
            self._rl[prefix](rl)
            self.log("{prefix}/classification_loss", self._cl[prefix], on_epoch=True)
            self.log("{prefix}/robust_loss", self._rl[prefix], on_epoch=True)

        return {
            "loss": cl + hparams.lbd * rl,
            "predictions": pred_softmax,
        }
