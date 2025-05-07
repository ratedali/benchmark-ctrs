from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal
from typing_extensions import TypedDict, Unpack, override

from benchmark_ctrs.training.methods.abc import (
    Batch,
    BatchResults,
    TrainingMethod,
)
from benchmark_ctrs.training.metrics import ScalarTags
from benchmark_ctrs.training.parameters import TrainingParameters


class CustomParameters(TrainingParameters):
    max_m: int
    lbd: float
    beta: float
    gamma: float
    deferred: bool


class _LossResult(TypedDict):
    loss: torch.Tensor
    cl: torch.Tensor
    rl: torch.Tensor | None
    pred: torch.Tensor


class _ExtraMetrics:
    def __init__(self, cl: float, rl: float | None):
        self.cl = cl
        self.rl = rl

    def get_scalars(self):
        scalars = {
            ScalarTags.Loss.Classification: self.cl,
        }
        if self.rl is not None:
            scalars[ScalarTags.Loss.Robust] = self.rl
        return scalars


def _make_batch_results(loss_result: _LossResult):
    return BatchResults(
        predictions=loss_result["pred"],
        loss=loss_result["loss"],
        extra_metrics=_ExtraMetrics(
            cl=loss_result["cl"].mean().item(),
            rl=loss_result["rl"].mean().item()
            if loss_result["rl"] is not None
            else None,
        ),
    )


class CustomTraining(TrainingMethod[CustomParameters]):
    norm = Normal(0.0, 1.0)

    def __init__(self, **params: Unpack[CustomParameters]):
        self.params = params
        self.skip_rl = self.params["deferred"]

    @staticmethod
    @override
    def create_instance(params):
        return CustomTraining(**params)

    @property
    @override
    def instance_tag(self):
        p = self.params
        return (
            "custom_var_deferred" if p["deferred"] else "custom_var",
            f"max_{p['max_m']}",
            f"lambda_{p['lbd']}",
            f"beta_{p['beta']}_gamma_{p['gamma']}",
            "adre",
        )

    def train(self, ctx, batch):
        # defer calculating the robust loss until the first learning rate change
        self.skip_rl = self.skip_rl and (
            ctx.scheduler is not None
            and np.array_equal(ctx.scheduler.base_lrs, ctx.scheduler.get_last_lr())
        )

        loss = self._loss(
            batch,
            model=ctx.model_wrapper.model,
            criterion=ctx.criterion,
            sigma=ctx.noise_sd,
            skip_rl=self.skip_rl,
            device=ctx.device,
            lbd=self.params["lbd"],
            max_m=self.params["max_m"],
            beta=self.params["beta"],
            gamma=self.params["gamma"],
        )

        # compute gradient and do SGD step
        ctx.optimizer.zero_grad()
        loss["loss"].mean().backward()
        ctx.optimizer.step()

        return _make_batch_results(loss)

    @override
    def test(self, ctx, batch):
        loss = self._loss(
            batch,
            model=ctx.model_wrapper.model,
            criterion=ctx.criterion,
            sigma=ctx.noise_sd,
            skip_rl=False,
            device=ctx.device,
            lbd=self.params["lbd"],
            max_m=self.params["max_m"],
            beta=self.params["beta"],
            gamma=self.params["gamma"],
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
        beta: float,
        gamma: float,
        lbd: float,
        max_m: int,
        skip_rl: bool = False,
        device: torch.device | None = None,
    ):
        # dim BxCxWxH
        samples = batch.inputs + torch.randn_like(batch.inputs, device=device) * sigma
        # dim BxK
        logits: torch.Tensor = model(samples)
        cl = criterion(logits, batch.targets)

        rl = torch.zeros_like(cl, device=device)
        if not skip_rl and lbd != 0:
            with torch.inference_mode():
                top1_pred = torch.topk(logits, k=1)
                correct = top1_pred.indices[:, 0] == batch.targets
                scores, selected_idx = torch.topk(
                    top1_pred.values[correct, :].flatten(),
                    k=min(max_m, int(correct.sum().item())),
                )
                freq = torch.ceil(max_m * torch.softmax(scores, dim=0)).int()
                selected_inputs = batch.inputs[correct][selected_idx]
                # dim (Bxm)xCxWxH, where m = floor(conf * max_)
                # each of the selected input sample is repeated
                # with a frequency proportional to the confidence
                rl_samples = selected_inputs.repeat_interleave(freq, dim=0)
                rl_samples += torch.randn_like(rl_samples, device=device) * sigma
            beta_logits = model(rl_samples) * beta
            beta_softmax = torch.softmax(beta_logits, dim=1)
            beta_softmax = torch.nested.nested_tensor_from_jagged(
                beta_softmax, lengths=freq
            ).mean(dim=1)
            beta_top2 = torch.topk(beta_softmax, 2)
            pA, pB = beta_top2.values[:, 0], beta_top2.values[:, 1]
            with torch.no_grad():
                zeta_tmp = torch.zeros_like(pA)
                zeta_tmp = cls.norm.icdf(pA) - cls.norm.icdf(pB)
                # apply hinge and discard nan and inf values
                nonzero = (
                    ~torch.isnan(zeta_tmp)
                    & ~torch.isinf(zeta_tmp)
                    & (torch.abs(zeta_tmp) <= gamma)
                )
            zeta = cls.norm.icdf(pA[nonzero]) - cls.norm.icdf(pB[nonzero])
            rl[correct][selected_idx][nonzero] = sigma * (gamma - zeta) / 2

        return _LossResult(
            cl=cl,
            rl=rl,
            loss=cl + lbd * rl,
            pred=logits,
        )
