from __future__ import annotations

import torch
from torch.nn import functional as F
from typing_extensions import TypedDict, Unpack, override

from benchmark_ctrs.training.methods.abc import (
    Batch,
    BatchResults,
    TrainingMethod,
)
from benchmark_ctrs.training.metrics import ScalarTags
from benchmark_ctrs.training.parameters import TrainingParameters


class ADREParameters(TrainingParameters):
    k: int
    lbd: float
    adversarial: bool


class _LossResult(TypedDict):
    R: torch.Tensor
    R_per: torch.Tensor
    R_adre: torch.Tensor
    pred: torch.Tensor


class _ADREExtraMetrics:
    def __init__(self, R_per: float, R_adre: float):
        self.R_per = R_per
        self.R_adre = R_adre

    def get_scalars(self):
        return {
            ScalarTags.Loss.Classification: self.R_per,
            ScalarTags.Loss.Robust: self.R_adre,
        }


def _make_batch_results(loss_result: _LossResult):
    return BatchResults(
        predictions=loss_result["pred"],
        loss=loss_result["R"],
        extra_metrics=_ADREExtraMetrics(
            R_per=loss_result["R_per"].mean().item(),
            R_adre=loss_result["R_adre"].mean().item(),
        ),
    )


class ADRETraining(TrainingMethod[ADREParameters]):
    """Implements ADRE regularized training from the paper:
    "Regularized Training and Tight Certification for Randomized Smoothed Classifier with Provable Robustness", Feng et al., 2020
    """  # noqa: E501

    def __init__(self, **params: Unpack[ADREParameters]):
        self.params = params

    @staticmethod
    @override
    def create_instance(params):
        return ADRETraining(**params)

    @property
    @override
    def instance_tag(self):
        p = self.params
        return (
            "adre_adv" if p["adversarial"] else "adre",
            f"m_{p['k']}",
            f"lambda_{p['lbd']}",
        )

    def train(self, ctx, batch):
        loss = ADRETraining._loss(
            batch,
            model=ctx.model_wrapper.model,
            criterion=ctx.criterion,
            sigma=ctx.noise_sd,
            device=ctx.device,
            lbd=self.params["lbd"],
            k=self.params["k"],
        )

        # compute gradient and do SGD step
        ctx.optimizer.zero_grad()
        loss["R"].mean().backward()
        ctx.optimizer.step()

        return _make_batch_results(loss)

    @override
    def test(self, ctx, batch):
        loss = ADRETraining._loss(
            batch,
            model=ctx.model_wrapper.model,
            criterion=ctx.criterion,
            sigma=ctx.noise_sd,
            device=ctx.device,
            lbd=self.params["lbd"],
            k=self.params["k"],
        )
        return _make_batch_results(loss)

    @classmethod
    def _loss(
        cls,
        batch: Batch,
        *,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        sigma: float,
        device: torch.device | None = None,
        lbd: float,
        k: int,
    ):
        samples = torch.repeat_interleave(batch.inputs, k, dim=0)  # (B*k)xCxWxH
        samples += torch.randn_like(samples, device=device) * sigma

        sample_logits: torch.Tensor = model(samples)  # (B*k)xK
        sample_targets = torch.repeat_interleave(batch.targets, k)

        R_per_samples: torch.Tensor = criterion(
            sample_logits,
            sample_targets,
        )  # (B*k)x1
        R_per = torch.unflatten(R_per_samples, dim=0, sizes=(-1, k)).mean(dim=1)  # Bx1

        if lbd == 0:
            # input_probs will only need gradients if robust loss is also needed
            sample_logits = sample_logits.detach()

        # smoothed class probabilities \hat{G}(x), shape same as batch.targets
        input_probs = (
            torch.softmax(sample_logits, dim=1)
            .unflatten(dim=0, sizes=(-1, k))
            .mean(dim=1)
        )

        R_adre = torch.zeros_like(R_per, device=device)
        if lbd != 0:
            class_logprobs = torch.log(input_probs + 1e-10)  # avoid NaN results
            top2 = torch.topk(class_logprobs, 2)
            use_c2 = (
                top2.indices[:, 0] == batch.targets
            )  # argmax_{c} \hat{G}^c(x) ==  y_i, thus use second top class

            R_adre[use_c2] = -F.nll_loss(
                class_logprobs[use_c2],
                top2.indices[use_c2, 1],
                reduction="none",
            )
            R_adre[~use_c2] = -F.nll_loss(
                class_logprobs[~use_c2],
                top2.indices[~use_c2, 0],
                reduction="none",
            )

        return _LossResult(
            R_per=R_per,
            R_adre=R_adre,
            R=R_per + lbd * R_adre,
            pred=input_probs,
        )
