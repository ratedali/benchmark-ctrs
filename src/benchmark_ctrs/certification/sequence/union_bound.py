from typing import cast

import numpy as np
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
from typing_extensions import override

from benchmark_ctrs.certification.sequence._utils import RunningTrial
from benchmark_ctrs.certification.sequence.base import (
    SequenceCertification,
)
from benchmark_ctrs.types import Batch, Classifier

__all__ = ["UBCertification"]


class UBCertification(SequenceCertification[RunningTrial]):
    @override
    def pre_certify(
        self,
        model: Classifier,
        data: Batch,
        sigma: float,
        alpha: float,
    ) -> None:
        super().pre_certify(model, data, sigma, alpha)
        alpha_t = _alpha_t(alpha, self.n)
        t = len(alpha_t)
        while t > 0:
            t -= 1
            if alpha_t[t] > 0:
                break
        self.alpha_t = alpha_t[:t]

    @override
    def empty_trial(self) -> RunningTrial:
        return RunningTrial(0, 0)

    @override
    def update_trial(
        self,
        trial: RunningTrial,
        pred: int,
        y: int,
        alpha: float,
    ) -> RunningTrial:
        A = trial.countA
        N = trial.num_samples

        if len(self.alpha_t) <= N:
            return trial.mark_done()

        alpha_t = self.alpha_t[N]
        if alpha_t > 0:
            pA = _lower_conf_bound(A, N, alpha_t)
            return trial.update_pA(pA)

        return trial


def _lower_conf_bound(x: int, n: int, alpha: float) -> float:
    ci_low = proportion_confint(x, n, alpha=2 * alpha, method="beta")[0]
    return cast("float", ci_low)


def _alpha_t(alpha: float, n: int = 100_000, beta: float = 1.1, kinit: int = 11):
    k = kinit
    alpha_t = [0.0] * (n + 1)
    for t in range(1, n):
        if t > beta**k:
            k += 1
            k_t = k - kinit
            alpha_t[t] = 5 * alpha / (k_t + 4) / (k_t + 5)
    return alpha_t


def _ub_thresholds(alpha: float, targetp: float = 0.5, n: int = 100_000):
    alpha_t = {}

    def _alpha_t(k: int) -> float:
        return 5 * alpha / (k + 4) / (k + 5)

    def upper_threshold(p: float):
        beta = 1.1
        kinit = 11
        k = kinit
        up = np.arange(n) + 100

        for t in range(1, n):
            if t > beta**k:
                k += 1
                alpha_t[t] = _alpha_t(k - kinit)
                up[t] = binom.ppf(1 - alpha_t[t], t, p) + 1

        cumin = np.max(up)
        for t in range(len(up))[::-1]:
            cumin = min(up[t], cumin)
            up[t] = cumin

        return up

    hi = upper_threshold(targetp)
    lo = hi if targetp == 0.5 else upper_threshold(1 - targetp)  # noqa: PLR2004
    lo = np.maximum.accumulate([i - j for i, j in enumerate(lo)])
    return list(lo), list(hi), alpha_t
