import math
from typing import Literal, NamedTuple, Optional, cast

import numpy as np
import torch
from lightning.pytorch.utilities import LightningEnum
from scipy.stats import binom, norm
from statsmodels.stats.proportion import proportion_confint
from torch import Tensor
from typing_extensions import override

from benchmark_ctrs.certification.abc import (
    _ABSTAIN,
    Certificate,
    CertificationMethod,
)
from benchmark_ctrs.certification.utils import BatchIndex, InputId, SamplingQueue
from benchmark_ctrs.types import Batch, Classifier


class _Trial(NamedTuple):
    countA: int
    num_samples: int
    pA: Optional[float]

    def with_sample(self, correct: bool) -> "_Trial":
        return _Trial(
            countA=self.countA + correct,
            num_samples=self.num_samples + 1,
            pA=self.pA,
        )

    def with_result(self, pA: float) -> "_Trial":
        return _Trial(
            countA=self.countA,
            num_samples=self.num_samples,
            pA=min(self.pA, pA) if self.pA is not None else pA,
        )


SequenceModeOptions = Literal["union-bound", "betting"]


class SequenceMode(LightningEnum):
    UnionBound = "union-bound"
    Betting = "betting"


class SequenceCertificaiton(CertificationMethod):
    def __init__(
        self,
        n0: int = 128,
        n: int = 100_000,
        batch_size: int = 10_000,
        mode: SequenceModeOptions = "union-bound",
        early_stopping: float = 1e-4,
    ) -> None:
        super().__init__()
        parsed_mode = SequenceMode.try_from_str(mode, source="value")
        if mode is None:
            raise ValueError(
                f'Got unknown value "{mode}" for  `mode`.'
                f" Supported values are {[m.value for m in SequenceMode]}"
            )

        self.n0 = n0
        self.n = n
        self.batch_size = batch_size
        self.mode = cast("SequenceMode", parsed_mode)
        self.early_stopping = early_stopping

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
        return self.predict_batch(
            model,
            x.unsqueeze(dim=0),
            sigma=sigma,
            alpha=alpha,
            num_classes=num_classes,
        )[0]

    @override
    def predict_batch(
        self,
        model: Classifier,
        inputs: Tensor,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> list[Certificate]:
        targets = self._estimate_cA(model, inputs, sigma)
        batch: Batch = (inputs, targets)
        return self.certify_batch(
            model,
            batch,
            sigma=sigma,
            alpha=alpha,
            num_classes=num_classes,
        )

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
        batch = (
            x.unsqueeze(dim=0),
            torch.tensor([y], dtype=torch.long, device=x.device),
        )
        return self.certify_batch(
            model,
            batch,
            sigma=sigma,
            alpha=alpha,
            num_classes=num_classes,
        )[0]

    @override
    def certify_batch(
        self,
        model: Classifier,
        data: Batch,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> list[Certificate]:
        if self.mode == SequenceMode.UnionBound:
            return self._ub_certify(model, data, sigma=sigma, alpha=alpha)
        raise ValueError("Sequence Mode is not yet implemented")

    def _ub_certify(
        self,
        model: Classifier,
        data: Batch,
        *,
        sigma: float,
        alpha: float = 0.001,
    ) -> list[Certificate]:
        queue = SamplingQueue(self.batch_size, data)
        trials: dict[InputId, _Trial] = dict.fromkeys(
            queue.input_ids, _Trial(0, 0, None)
        )
        X, y = queue.batch

        lo, hi, alphas = self._ub_thresholds(alpha, self.n)
        results: dict[InputId, Certificate] = {}
        done = 0
        while done < queue.total:
            preds = self._sample_noise(model, X, sigma).argmax(-1)
            preds = cast("list[bool]", (preds == y).tolist())
            for i, correct in enumerate(preds):
                batch_idx = BatchIndex(i)
                if queue.new[batch_idx]:
                    queue.new[batch_idx] = False
                    continue

                input_id = queue.batch_ids[batch_idx]
                trial = trials[input_id].with_sample(correct)
                trials[input_id] = trial

                if (trial.num_samples >= len(lo) and input_id not in results) or (
                    trial.num_samples < len(lo)
                    and lo[trial.num_samples] == trial.countA
                ):
                    results[input_id] = Certificate(_ABSTAIN, 0)
                elif (
                    trial.num_samples in alphas
                    and trial.countA >= hi[trial.num_samples]
                ):
                    alpha_t = alphas[trial.num_samples]
                    pA_current = trial.pA
                    pA_new = self._lower_conf_bound(
                        trial.countA,
                        trial.num_samples,
                        alpha_t,
                    )
                    trial = trial.with_result(pA_new)
                    trials[input_id] = trial

                    if pA_current is not None and math.isclose(
                        pA_new,
                        pA_current,
                        abs_tol=self.early_stopping,
                    ):
                        radius = self._cert_radius(sigma, pA_new)
                        cA = int(y[batch_idx].item())
                        results[input_id] = Certificate(cA, radius, trial.pA)

                if input_id in results:
                    done += 1
                    if done == queue.total:
                        break
                    queue.replace_input(input_id, exclude=results)

        return [results[id_] for id_ in queue.input_ids]

    def _estimate_cA(self, model: Classifier, inputs: Tensor, sigma: float) -> Tensor:
        trials: dict[InputId, tuple[int, list[int]]] = {}
        results: dict[InputId, int] = {}
        queue = SamplingQueue(self.batch_size, inputs)

        done = 0
        while done < queue.total:
            preds = self._sample_noise(model, inputs, sigma)
            num_classes = preds.size(-1)
            preds = preds.argmax(dim=-1).tolist()

            for i, pred in enumerate(preds):
                batch_idx = BatchIndex(i)
                if queue.new[batch_idx]:
                    queue.new[batch_idx] = False
                    continue

                id_ = queue.batch_ids[batch_idx]
                num, counts = trials.get(id_, (0, [0] * num_classes))
                num += 1
                counts[pred] += 1
                trials[id_] = (num, counts)

                if num == self.n0:
                    results[id_] = int(np.argmax(counts))
                    done += 1
                    if done == queue.total:
                        break
                    queue.replace_input(id_, exclude=results)
        return torch.tensor(
            [results[id_] for id_ in queue.input_ids],
            dtype=torch.long,
            device=inputs.device,
        )

    def _sample_noise(
        self,
        model: Classifier,
        inputs: Tensor,
        sigma: float,
    ) -> Tensor:
        noise = torch.randn_like(inputs) * sigma
        return model(inputs + noise)

    def _cert_radius(
        self,
        sigma: float,
        pA: float,
        pB: Optional[float] = None,
    ) -> float:
        if pB is None:
            return sigma * cast("float", norm.ppf(pA).item())
        return sigma * cast("float", norm.ppf(pA) - norm.ppf(pB)) / 2

    def _lower_conf_bound(self, x: int, n: int, alpha: float) -> float:
        ci_low = proportion_confint(x, n, alpha=2 * alpha, method="beta")[0]
        return cast("float", ci_low)

    def _ub_thresholds(
        self,
        alpha: float,
        n: int = 100_000,
    ) -> tuple[list[int], list[int], dict[int, float]]:
        alpha_t = {}

        def _alpha_t(k: int) -> float:
            return 5 * alpha / (k + 4) / (k + 5)

        def upper_threshold(p: float, n: int):
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

        hi = upper_threshold(0.5, n)
        lo = np.maximum.accumulate([i - j for i, j in enumerate(hi)])
        return list(lo), list(hi), alpha_t
