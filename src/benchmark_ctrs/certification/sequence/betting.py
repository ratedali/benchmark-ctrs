import dataclasses
import logging
from math import log
from typing import cast

import numpy as np
from scipy.optimize import bisect
from typing_extensions import Self, override

from benchmark_ctrs.certification.sequence._utils import RunningTrial
from benchmark_ctrs.certification.sequence.base import (
    SequenceCertification,
)

__all__ = ["BettingCertification"]

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BettingTrial(RunningTrial):
    logQ: float = 0

    def update_logQ(self, logQ: float) -> "Self":
        return dataclasses.replace(self, logQ=self.logQ + logQ)


class BettingCertification(SequenceCertification[BettingTrial]):
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
    ) -> BettingTrial:
        H = trial.countA
        t = trial.num_samples

        x = pred == y
        q_hat = (H + 0.5) / (t + 1)
        trial = trial.update_logQ(x * log(q_hat) + (1 - x) * log(1 - q_hat))

        logW = _logW(trial.logQ, H, t, alpha)
        a = max(1e-10, trial.pA)
        b = min(H / t, 1 - 1e-10)
        if a < b and logW(a) >= 0 and logW(b) < 0:
            try:
                pA = cast(
                    "float",
                    bisect(
                        f=logW,
                        a=a,
                        b=b,
                        xtol=1e-8,
                    ),
                )
                return trial.update_pA(pA)
            except ValueError as err:
                _logger.warning(
                    "Error when updating the CI, no values for p are viable"
                    " betwen (a = %.3e, f(a) = %.3e) and (b = %.3e, f(b) = %.3e)."
                    " (%s)",
                    a,
                    logW(a),
                    b,
                    logW(b),
                    str(err),
                )
        return trial


def _logW(logQ: float, H: int, t: int, alpha: float):
    def impl(p: float):
        logP = H * log(p) + (t - H) * log(1 - p)
        return logQ - logP - log(1 / (2 * alpha))

    return impl


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
