from math import ceil
from typing import Literal, cast

import numpy as np
import torch
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from torch import Tensor
from typing_extensions import override

from benchmark_ctrs.certification import (
    _ABSTAIN,
    Certificate,
    CertificationMethod,
)
from benchmark_ctrs.types import Classifier


class RSCertification(CertificationMethod):
    def __init__(
        self,
        n0: int = 100,
        n: int = 100_000,
        batch_size: int = 10_000,
        mode: Literal["gaussian", "cauchy"] = "gaussian",
        ord_: Literal["l1", "l2", "linf"] = "l2",
    ) -> None:
        """

        Args:
            num_classes (int): number of classes in the dataset.
            n0 (int, optional): number of samples for top class estimation.
                Defaults to 100.
            n (int, optional): number of samples for certified radius CI estimation.
                Defaults to 100_000.
            batch_size (int, optional): Batch size used for sampling.
                Defaults to 10_000.
            mode (Literal[&quot;gaussian&quot;, &quot;cauchy&quot;], optional):
              attack noise type. Defaults to "gaussian".
            ord_ (Literal[&quot;l1&quot;, &quot;l2&quot;, &quot;linf&quot;], optional):
                Defaults to "l2".
        """

        super().__init__()
        self.n0 = n0
        self.n = n
        self.bs = batch_size
        self.mode = mode
        self.ord = ord_

    @override
    def certify(
        self,
        model: Classifier,
        x: Tensor,
        y: int,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> Certificate:
        return self._certify_cohen(
            model,
            x,
            y,
            sigma=sigma,
            alpha=alpha,
            num_classes=num_classes,
        )

    @override
    def predict(
        self,
        model: Classifier,
        x: Tensor,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> Certificate:
        cA = self._estimate_cA(
            model,
            x,
            sigma=sigma,
            num_classes=num_classes,
        )
        return self._certify_cohen(
            model,
            x,
            cA,
            sigma=sigma,
            alpha=alpha,
            num_classes=num_classes,
        )

    def _estimate_cA(
        self,
        model: Classifier,
        x: Tensor,
        *,
        sigma: float,
        num_classes: int,
    ) -> int:
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(
            model,
            x,
            sigma=sigma,
            num=self.n0,
            batch_size=self.bs,
            num_classes=num_classes,
        )
        # use these samples to take a guess at the top class
        return int(counts_selection.argmax().item())

    def _certify_cohen(
        self,
        model: Classifier,
        x: Tensor,
        cA: int,
        *,
        sigma: float,
        alpha: float,
        num_classes: int,
    ) -> Certificate:
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(
            model,
            x,
            sigma=sigma,
            num=self.n,
            batch_size=self.bs,
            num_classes=num_classes,
        )
        # use these samples to estimate a lower bound on pA
        nA = cast("int", counts_estimation[cA].item())
        pABar = self._lower_confidence_bound(nA, self.n, alpha)
        if pABar < 0.5:  # noqa: PLR2004
            return Certificate(_ABSTAIN, 0.0)

        if self.mode == "gaussian":
            radius = sigma * cast("float", norm.ppf(pABar).item())
            if self.ord == "l2":
                pass
            elif self.ord == "linf":
                radius = radius / cast("float", np.sqrt(x.numel()).item())
            else:
                raise NotImplementedError
        elif self.mode == "cauchy":
            alpha = (4 * pABar * (1 - pABar)) ** (-1 / (2 * x.numel()))
            radius = 2 * sigma * cast("float", np.sqrt(alpha - 1).item())
            if self.ord == "linf":
                pass
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return Certificate(cA, radius)

    def _sample_noise(
        self,
        model: Classifier,
        x: Tensor,
        *,
        sigma: float,
        num: int,
        batch_size: int,
        num_classes: int,
    ) -> Tensor:
        """Sample the base classifier's prediction under noisy corruptions
        of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: a scalar tensor of length num_classes containing the per-class counts
        """
        counts = torch.zeros(size=(num_classes,), dtype=torch.int, device=x.device)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))

            if self.mode == "gaussian":
                noise = torch.randn_like(batch) * sigma
            elif self.mode == "cauchy":
                noise = torch.empty_like(batch).cauchy_() * sigma
            else:
                raise NotImplementedError
            predictions = model(batch + noise).argmax(1)
            counts += predictions.bincount(minlength=num_classes)
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
