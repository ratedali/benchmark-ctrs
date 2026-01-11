from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, cast

import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor
from typing_extensions import override

from benchmark_ctrs.certification.abc import _ABSTAIN, Certificate, CertificationMethod
from benchmark_ctrs.certification.sequence._utils import (
    BatchIndex,
    InputId,
    RunningTrial,
    SamplingQueue,
)
from benchmark_ctrs.types import Batch, Classifier

_TTrial = TypeVar("_TTrial", bound=RunningTrial)


class SequenceCertification(CertificationMethod, ABC, Generic[_TTrial]):
    def __init__(self, n0: int, n: int, batch_size: int) -> None:
        super().__init__()
        self.n0 = n0
        self.n = n
        self.batch_size = batch_size

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
        targets = self.estimate_cA(model, inputs, sigma)
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

    def pre_certify(
        self,
        model: Classifier,
        data: Batch,
        sigma: float,
        alpha: float,
    ) -> None: ...

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
        self.pre_certify(model=model, data=data, sigma=sigma, alpha=alpha)

        queue = SamplingQueue(self.batch_size, data)
        trials = dict.fromkeys(
            queue.input_ids,
            self.empty_trial(),
        )
        X, y = queue.batch

        results: dict[InputId, Certificate] = {}
        done = 0
        while done < queue.total:
            preds = self.sample_noise(model, X, sigma).argmax(-1)
            preds = cast("list[bool]", (preds == y).tolist())
            for i, correct in enumerate(preds):
                batch_idx = BatchIndex(i)
                if queue.new[batch_idx]:
                    queue.new[batch_idx] = False
                    continue

                input_id = queue.batch_ids[batch_idx]
                trial = trials[input_id].add_sample(correct=correct)

                update = self.update_trial(trial, alpha)
                if update is not None:
                    trials[input_id] = update

                if trials[input_id].done or trials[input_id].num_samples >= self.n:
                    cA = int(y[batch_idx].item())
                    results[input_id] = self.get_certificate(
                        trials[input_id],
                        cA,
                        sigma,
                    )

                if input_id in results:
                    done += 1
                    if done == queue.total:
                        break
                    queue.replace_input(input_id, exclude=results)

        return [results[id_] for id_ in queue.input_ids]

    @abstractmethod
    def empty_trial(self) -> _TTrial: ...

    @abstractmethod
    def update_trial(
        self,
        trial: _TTrial,
        alpha: float,
    ) -> Optional[_TTrial]: ...

    def estimate_cA(self, model: Classifier, inputs: Tensor, sigma: float) -> Tensor:
        trials: dict[InputId, tuple[int, list[int]]] = {}
        results: dict[InputId, int] = {}
        queue = SamplingQueue(self.batch_size, inputs)

        done = 0
        while done < queue.total:
            preds = self.sample_noise(model, inputs, sigma)
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

    def sample_noise(
        self,
        model: Classifier,
        inputs: Tensor,
        sigma: float,
    ) -> Tensor:
        noise = torch.randn_like(inputs) * sigma
        return model(inputs + noise)

    def get_certificate(
        self,
        trial: _TTrial,
        y: int,
        sigma: float,
    ) -> Certificate:
        pA = trial.pA
        if pA is None:
            return Certificate(prediction=_ABSTAIN, radius=0)
        radius = sigma * cast("float", norm.ppf(pA).item())
        return Certificate(prediction=y, radius=radius, pA=pA)
