from typing import Optional, cast

import numpy as np
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint
from typing_extensions import override

from benchmark_ctrs.certification.sequence._utils import RunningTrial
from benchmark_ctrs.certification.sequence.base import (
    SequenceCertification,
)
from benchmark_ctrs.types import Batch, Classifier


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
        self.lo, self.hi, self.alphas = _ub_thresholds(alpha=alpha, n=self.n)

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
    ) -> Optional[RunningTrial]:
        A = trial.countA
        N = trial.num_samples
        alpha_t = self.alphas.get(N)

        if self.n <= N:
            return trial.mark_done()

        if self.lo[N] == A:
            return trial.update_pA(None).mark_done()

        if alpha_t is not None and self.hi[N] <= A:
            pA = _lower_conf_bound(A, N, alpha_t)
            return trial.update_pA(pA)

        return None


def _lower_conf_bound(x: int, n: int, alpha: float) -> float:
    ci_low = proportion_confint(x, n, alpha=2 * alpha, method="beta")[0]
    return cast("float", ci_low)


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
