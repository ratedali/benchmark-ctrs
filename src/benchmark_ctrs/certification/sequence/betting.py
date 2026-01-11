import dataclasses
from math import ceil, log
from typing import Optional, cast

import numpy as np
from scipy.optimize import bisect
from typing_extensions import Self, override

from benchmark_ctrs.certification.sequence._utils import RunningTrial
from benchmark_ctrs.certification.sequence.base import (
    SequenceCertification,
)
from benchmark_ctrs.types import Batch, Classifier


@dataclasses.dataclass(frozen=True)
class BettingTrial(RunningTrial):
    logQ: float = 0

    def update_logQ(self, logQ: float) -> "Self":
        return dataclasses.replace(
            self,
            logQ=logQ,
        )


class BettingCertification(SequenceCertification[BettingTrial]):
    @override
    def pre_certify(
        self,
        model: Classifier,
        data: Batch,
        sigma: float,
        alpha: float,
    ) -> None:
        self.update_steps = _ci_update_steps(n=self.n)
        self.lo, self.hi = _betting_thresholds(alpha, n=self.n)

    @override
    def empty_trial(self) -> BettingTrial:
        return BettingTrial(0, 0)

    @override
    def update_trial(
        self,
        trial: BettingTrial,
        pred: int,
        y: int,
        alpha: float,
    ) -> Optional[BettingTrial]:
        H = trial.countA
        t = trial.num_samples

        if self.n <= t:
            return trial.mark_done()

        if self.lo[t] == H:
            return trial.update_pA(None).mark_done()

        x = pred == y
        q_hat = (H + 0.5) / (t + 1)
        logQ = trial.logQ + x * log(q_hat) + (1 - x) * log(1 - q_hat)
        update = trial.update_logQ(logQ)

        if self.update_steps[t] and self.hi[t] <= H:
            tol = 1e-8
            pA = cast(
                "float",
                bisect(
                    f=_logW(trial.logQ, H, t, alpha),
                    a=min(H / t, 1 - tol),
                    b=tol,
                    xtol=tol,
                ),
            )
            return update.update_pA(pA)
        return update


def _logW(logQ: float, H: int, t: int, alpha: float):
    def compute(p: float):
        logP = H * log(p) + (t - H) * log(1 - p)
        return logQ - logP - log(1 / (2 * alpha))

    return compute


def _ci_update_steps(beta: float = 1.1, kinit: int = 11, n: int = 100_000):
    should_update = np.zeros(n + 1, dtype=np.bool)
    kmax = ceil(log(n, beta))
    powers = np.arange(kinit, kmax)
    update_at = np.ceil(np.pow(beta, powers)).astype(np.long)
    should_update[update_at] = True
    should_update[n] = True
    return should_update


def _betting_thresholds(alpha: float, targetp: float = 0.5, n: int = 100_000):
    def upper_threshold(p: float):
        thresholds = [0] * (n + 1)

        h = 0
        t = 0
        logQ = 0

        while t <= n:
            H = h
            for h in range(H, t + 1):
                logP = h * log(p) + (t - h) * log(1 - p)
                if (logQ - logP > log(1 / alpha / 2)) and h >= p * t:
                    thresholds[t] = h
                    t += 1
                    q = (h + 0.5) / t
                    logQ += log(1 - q)
                    break

                if h != t:
                    q = (h + 0.5) / t
                    logQ = logQ - log(1 - q) + log(q)
                else:
                    thresholds[t] = n + h
                    t += 1
                    q = (h + 0.5) / t
                    logQ += log(1 - q)

        cumin = n * 2
        for i in range(len(thresholds))[::-1]:
            cumin = min(thresholds[i], cumin)
            thresholds[i] = cumin

        return thresholds

    hi = upper_threshold(targetp)
    lo = hi if targetp == 0.5 else upper_threshold(1 - targetp)  # noqa: PLR2004
    lo = np.maximum.accumulate([t - h for t, h in enumerate(lo)])
    return list(lo), list(hi)
