# this file is based on code available publically on
#   https://github.com/locuslab/smoothing
# originally written by Jeremy Cohen.
from __future__ import annotations

import dataclasses
from math import ceil
from typing import TYPE_CHECKING, NamedTuple, cast, overload

import lightning as L
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import binomtest, norm
from statsmodels.stats.proportion import proportion_confint
from typing_extensions import TypeAlias, TypeIs

if TYPE_CHECKING:
    from typing import Literal

    from torch import Tensor

    from benchmark_ctrs.types import Classifier


class _ABSTAIN_TYPE: ...


_ABSTAIN = _ABSTAIN_TYPE()

Prediction: TypeAlias = "int | _ABSTAIN_TYPE"


def is_abstain(prediction: Prediction) -> TypeIs[_ABSTAIN_TYPE]:
    return prediction == _ABSTAIN


class Certificate(NamedTuple):
    prediction: Prediction
    certified_radius: float


@dataclasses.dataclass(frozen=True)
class HParams:
    """
    Arguments:
    :param num_classes:
    :param sigma: the noise level hyperparameter
    :param n0: the number of Monte Carlo samples to use for selection
    :param n: the number of Monte Carlo samples to use for estimation
    :param batch_size: batch size to use when evaluating the base classifier
    :param alpha: the failure probability
    """

    alpha: float = 0.001
    n0: int = 100
    n: int = 100_000
    batch_size: int = 10_000
    mode: Literal["gaussian", "cauchy"] = "gaussian"
    ord: Literal["l1", "l2", "linf"] = "l2"


class SmoothedClassifier(L.LightningModule):
    """A smoothed classifier g"""

    def __init__(
        self,
        base_classifier: Classifier,
        num_classes: int,
        sigma: float,
        params: HParams,
    ) -> None:
        """
        :param base_classifier: maps from
            [batch x channel x height x width] to [batch x num_classes]

        """
        super().__init__()

        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.alpha = params.alpha
        self.n0 = params.n0
        self.n = params.n
        self.batch_size = params.batch_size
        self.mode = params.mode
        self.ord = params.ord

    @overload
    def forward(self, x: Tensor, *, certify: Literal[False] = False) -> Prediction: ...
    @overload
    def forward(self, x: Tensor, *, certify: Literal[True]) -> Certificate: ...
    def forward(self, x: Tensor, *, certify: bool = False):
        if certify:
            return self.certify(x)
        return self.predict(x)

    def certify(self, x: Tensor) -> Certificate:
        """Monte Carlo algorithm for certifying that g's prediction around x is constant
        within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will
        equal g(x), and g's prediction will be robust within a L2 ball
        of radius R around x.

        :param x: the input [channel x height x width]
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, self.n0, self.batch_size)
        # use these samples to take a guess at the top class
        cAHat = cast("int", counts_selection.argmax().item())
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, self.n, self.batch_size)
        # use these samples to estimate a lower bound on pA
        nA = cast("int", counts_estimation[cAHat].item())
        pABar = self._lower_confidence_bound(nA, self.n, self.alpha)
        if pABar < 0.5:  # noqa: PLR2004
            return Certificate(_ABSTAIN, 0.0)

        if self.mode == "gaussian":
            radius = self.sigma * cast("float", norm.ppf(pABar).item())
            if self.ord == "l2":
                pass
            elif self.ord == "linf":
                radius = radius / cast("float", np.sqrt(x.numel()).item())
            else:
                raise NotImplementedError
        elif self.mode == "cauchy":
            alpha = (4 * pABar * (1 - pABar)) ** (-1 / (2 * x.numel()))
            radius = 2 * self.sigma * cast("float", np.sqrt(alpha - 1).item())
            if self.ord == "linf":
                pass
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return Certificate(cAHat, radius)

    def certify_norm(self, x: Tensor) -> Certificate:
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise_norm(x, self.n0, self.batch_size)
        # use these samples to take a guess at the top class
        cAHat = cast("int", counts_selection.argmax().item())
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise_norm(x, self.n, self.batch_size)
        # use these samples to estimate a lower bound on pA
        nA = cast("int", counts_estimation[cAHat].item())
        pABar = self._lower_confidence_bound(nA, self.n, self.alpha)
        if pABar < 0.5:  # noqa: PLR2004
            return Certificate(_ABSTAIN, 0.0)

        radius = self.sigma * cast("float", norm.ppf(pABar).item())
        if self.ord == "l2":
            pass
        elif self.ord == "linf":
            radius = radius / cast("float", np.sqrt(x.numel()).item())
        else:
            raise NotImplementedError

        return Certificate(cAHat, radius)

    def predict(self, x: Tensor) -> Prediction:
        """Monte Carlo algorithm for evaluating the prediction of g at x.
        With probability at least 1 - alpha, the class returned by this method
        will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :return: the predicted class, or ABSTAIN
        """
        counts = self._sample_noise(x, self.n, self.batch_size)
        top2 = counts.argsort(descending=True)[:2]
        count1 = cast("int", counts[top2[0]].item())
        count2 = cast("int", counts[top2[1]].item())

        if binomtest(count1, count1 + count2, p=0.5).pvalue > self.alpha:
            return _ABSTAIN

        return cast("int", top2[0].item())

    def confidence_top2(self, x: Tensor) -> tuple[Tensor, tuple[float, float], Tensor]:
        """Monte Carlo algorithm for evaluating the prediction of g at x.
        With probability at least 1 - alpha, the class returned by this method
        will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the top2 confidence indices, their values and all class confidences
        """
        counts = self._sample_noise(x, self.n, self.batch_size)
        confidences = counts / self.n
        top2 = confidences.argsort()[::-1][:2]
        p1 = confidences[top2[0]].item()
        p2 = confidences[top2[1]].item()

        return top2, (p1, p2), confidences

    def _sample_noise(self, x: Tensor, num: int, batch_size: int) -> Tensor:
        """Sample the base classifier's prediction under noisy corruptions
        of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: a scalar tensor of length num_classes containing the per-class counts
        """
        counts = torch.zeros(
            size=(self.num_classes,), dtype=torch.int, device=self.device
        )
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))

            if self.mode == "gaussian":
                noise = torch.randn_like(batch) * self.sigma
            elif self.mode == "cauchy":
                noise = torch.empty_like(batch).cauchy_() * self.sigma
            else:
                raise NotImplementedError
            predictions = self.base_classifier(batch + noise).argmax(1)
            counts += predictions.bincount(minlength=self.num_classes)
        return counts

    def _sample_noise_norm(self, x: Tensor, num: int, batch_size) -> Tensor:
        """Sample the base classifier's prediction under noisy corruptions
        of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """

        def _batch_stddev(x: Tensor):
            x_flat = x.flatten(1)
            return torch.sqrt(x_flat.var(dim=1, unbiased=False) + 1e-8)

        def _add_norm(x: Tensor, d: Tensor):
            y = x + d
            s = _batch_stddev(y)
            return y / (s.view(-1, 1, 1, 1) + 1e-8)

        counts = torch.zeros(size=(self.num_classes,), dtype=torch.int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))

            noise = torch.randn_like(batch) * self.sigma
            predictions = self.base_classifier(_add_norm(batch, noise)).argmax(1)
            counts += predictions.bincount(minlength=self.num_classes)
        return counts

    def _count_arr(self, arr: npt.NDArray, length: int) -> npt.NDArray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true
                 w.p at least (1 - alpha) over the samples
        """
        ci_low, _ci_upp = proportion_confint(NA, N, alpha=2 * alpha, method="beta")
        return cast("float", ci_low)
