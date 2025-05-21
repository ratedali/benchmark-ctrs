from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
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
class ADREHParams(HParams):
    learning_rate: float = 0.1
    lr_decay: float = 0.1
    lr_step: int = 50
    momentum: float = 0.9
    weight_decay: float = 1e-4

    k: int = 8
    lbd: float = 0.1
    adversarial: bool = False


class ADRE(RandomizedSmoothing):
    def __init__(
        self,
        *args,
        params: ADREHParams,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, params=params)
        self._Lper_train = MeanMetric()
        self._Lper_val = MeanMetric()
        self._Ladre_train = MeanMetric()
        self._Ladre_val = MeanMetric()

    @override
    def training_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._loss(batch, stage="train")

    @override
    def validation_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._loss(batch, stage="val")

    def _loss(
        self,
        batch: Batch,
        *,
        stage: Literal["train", "val"] | None = None,
    ) -> StepOutput:
        hparams = cast("ADREHParams", self.hparams)
        inputs, targets = batch

        with record_function("sampling"):
            samples = torch.repeat_interleave(inputs, hparams.k, dim=0)  # (B*k)xCxWxH
            sample_logits: Tensor = self(samples)  # (B*k)xK
            sample_targets = torch.repeat_interleave(targets, hparams.k)

        with record_function("classification_loss"):
            R_per: Tensor = self._criterion(
                sample_logits,
                sample_targets,
            )

        if hparams.lbd == 0:
            # input_probs will only need gradients if robust loss is also needed
            sample_logits = sample_logits.detach()

        with record_function("robust_loss"):
            # smoothed class probabilities \hat{G}(x), shape same as targets
            input_probs = (
                torch.softmax(sample_logits, dim=1)
                .unflatten(dim=0, sizes=(-1, hparams.k))
                .mean(dim=1)
            )

            R_adre = torch.tensor(0)
            if hparams.lbd != 0:
                class_logprobs = torch.log(input_probs + 1e-10)  # avoid NaN results
                top2 = torch.topk(class_logprobs, 2)
                use_c2 = (
                    top2.indices[:, 0] == targets
                )  # argmax_{c} \hat{G}^c(x) ==  y_i, thus use second top class

                R_adre = -F.nll_loss(
                    class_logprobs[use_c2],
                    top2.indices[use_c2, 1],
                    reduction="sum",
                )
                R_adre += -F.nll_loss(
                    class_logprobs[~use_c2],
                    top2.indices[~use_c2, 0],
                    reduction="sum",
                )
                R_adre = R_adre / targets.numel()

        if stage == "train":
            self._Lper_train(R_per)
            self._Ladre_train(R_adre)
            self.log("train/classification_loss", self._Lper_train, on_epoch=True)
            self.log("train/robust_loss", self._Ladre_train, on_epoch=True)
        elif stage == "val":
            self._Lper_val(R_per)
            self._Ladre_val(R_adre)
            self.log("val/classification_loss", self._Lper_val, on_epoch=True)
            self.log("val/robust_loss", self._Ladre_val, on_epoch=True)

        return {"loss": R_per + hparams.lbd * R_adre, "predictions": input_probs}
