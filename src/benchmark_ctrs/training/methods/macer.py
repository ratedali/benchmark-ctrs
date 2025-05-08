from __future__ import annotations

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.profiler import record_function
from typing_extensions import TypedDict, Unpack, override

from benchmark_ctrs.training.methods.abc import (
    Batch,
    BatchResults,
    TrainingMethod,
)
from benchmark_ctrs.training.metrics import ScalarTags
from benchmark_ctrs.training.parameters import TrainingParameters


class MACERParameters(TrainingParameters):
    m_train: int
    lbd: float
    beta: float
    gamma: float
    deferred: bool


class _LossResult(TypedDict):
    cl: torch.Tensor
    rl: torch.Tensor
    pred: torch.Tensor


class _MACERExtraMetrics:
    def __init__(self, cl: float, rl: float):
        self.cl = cl
        self.rl = rl

    def get_scalars(self):
        return {
            ScalarTags.Loss.Classification: self.cl,
            ScalarTags.Loss.Robust: self.rl,
        }


def _make_batch_results(loss_result: _LossResult):
    return BatchResults(
        predictions=loss_result["pred"],
        loss=loss_result["cl"] + loss_result["rl"],
        extra_metrics=_MACERExtraMetrics(
            cl=loss_result["cl"].mean().item(),
            rl=loss_result["rl"].mean().item(),
        ),
    )


class MACERTraining(TrainingMethod[MACERParameters]):
    """Implements MACER regularized training from the paper:
    "MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius", Zhai et al., 2019
    """  # noqa: E501

    norm = Normal(0.0, 1.0)

    def __init__(self, **params: Unpack[MACERParameters]):
        self.params = params
        self.skipping_rl = self.params["deferred"]

    @staticmethod
    @override
    def create_instance(params):
        return MACERTraining(**params)

    @property
    @override
    def instance_tag(self):
        p = self.params
        return (
            "macer_deferred" if p["deferred"] else "macer",
            f"m_{p['m_train']}",
            f"lambda_{p['lbd']}",
            f"beta_{p['beta']}_gamma_{p['gamma']}",
        )

    def train(self, ctx, batch):
        # defer calculating the robust loss until the first learning rate change
        self.skipping_rl = self.skipping_rl and (
            ctx.scheduler is not None
            and np.array_equal(ctx.scheduler.base_lrs, ctx.scheduler.get_last_lr())
        )

        loss = self._loss(
            batch,
            ctx.model_wrapper.model,
            sigma=ctx.noise_sd,
            skip_rl=self.skipping_rl,
            device=ctx.device,
            lbd=self.params["lbd"],
            m_train=self.params["m_train"],
            gamma=self.params["gamma"],
            beta=self.params["beta"],
        )
        total_loss = loss["cl"] + loss["rl"]

        # compute gradient and do SGD step
        ctx.optimizer.zero_grad()
        with record_function("backprop"):
            total_loss.mean().backward()
        ctx.optimizer.step()

        with torch.inference_mode():
            return _make_batch_results(loss)

    @override
    def test(self, ctx, batch):
        loss = self._loss(
            batch,
            ctx.model_wrapper.model,
            sigma=ctx.noise_sd,
            skip_rl=False,
            device=ctx.device,
            lbd=self.params["lbd"],
            m_train=self.params["m_train"],
            gamma=self.params["gamma"],
            beta=self.params["beta"],
        )
        return _make_batch_results(loss)

    @classmethod
    def _loss(
        cls,
        batch: Batch,
        model: torch.nn.Module,
        *,
        sigma: float,
        skip_rl: bool = False,
        device: torch.device | None = None,
        lbd: float,
        m_train: int,
        gamma: float,
        beta: float,
    ):
        batch_size = batch.inputs.size(0)
        new_shape = [batch_size * m_train]
        new_shape.extend(batch.inputs[0].shape)

        with record_function("sampling"):
            noisy_inputs = batch.inputs.repeat((1, m_train, 1, 1)).view(new_shape)
            noisy_inputs += torch.randn_like(noisy_inputs, device=device) * sigma

            predictions = model(noisy_inputs)
            predictions = predictions.reshape((batch_size, m_train, -1))

        with record_function("classification_loss"):
            # perturbed loss
            pred_softmax = torch.softmax(predictions, dim=2).mean(1)
            pred_logsoftmax = torch.log(pred_softmax + 1e-10)  # avoid nan
            cl = F.nll_loss(pred_logsoftmax, batch.targets, reduction="none")

        with record_function("robust_loss"):
            rl = torch.zeros_like(cl, device=device)
            if not skip_rl and lbd != 0:
                # only apply beta to the robustness loss
                beta_pred = predictions * beta
                beta_pred_softmax = torch.softmax(beta_pred, dim=2).mean(1)

                top2 = torch.topk(beta_pred_softmax, 2)
                pA, pB = top2.values[:, 0], top2.values[:, 1]

                with torch.no_grad():
                    correct = top2.indices[:, 0] == batch.targets  # G_theta

                    zeta_tmp = torch.zeros_like(rl)
                    zeta_tmp[correct] = cls.norm.icdf(pA[correct]) - cls.norm.icdf(
                        pB[correct],
                    )
                    # apply hinge and discard nan and inf values
                    nonzero = (
                        correct
                        & ~torch.isnan(zeta_tmp)
                        & ~torch.isinf(zeta_tmp)
                        & (torch.abs(zeta_tmp) <= gamma)
                    )
                zeta = cls.norm.icdf(pA[nonzero]) - cls.norm.icdf(pB[nonzero])
                rl[nonzero] = lbd * sigma * (gamma - zeta) / 2
        return _LossResult(cl=cl, rl=rl, pred=pred_softmax)
