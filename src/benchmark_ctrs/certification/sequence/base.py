from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

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

__all__ = ["SequenceCertification"]

_TTrial = TypeVar("_TTrial", bound=RunningTrial)


class SequenceCertification(CertificationMethod, ABC, Generic[_TTrial]):
    def __init__(
        self,
        n0: int = 100,
        n: int = 100000,
        batch_size: int = 10000,
        *,
        early_stopping: bool = False,
        patience: int = 5000,
        mindelta: float = 0.0001,
    ) -> None:
        super().__init__()
        self.n0 = n0
        self.n = n
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.mindelta = mindelta

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
        alpha: float = 0.0001,
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
            preds = cast(
                "list[int]",
                self.sample_noise(model, X, sigma).argmax(-1).tolist(),
            )
            for i, pred in enumerate(preds):
                batch_idx = BatchIndex(i)
                if queue.new[batch_idx]:
                    queue.new[batch_idx] = False
                    continue

                cA = int(y[batch_idx].item())
                input_id = queue.batch_ids[batch_idx]

                trial = self.update_trial(
                    trials[input_id].add_sample(pred, cA),
                    pred=pred,
                    y=cA,
                    alpha=alpha,
                )

                if self.early_stopping:
                    trial = trial.check_stopping(self.patience, self.mindelta)

                trials[input_id] = trial

                if trial.done or trial.num_samples >= self.n:
                    results[input_id] = self.get_certificate(trial, y=cA, sigma=sigma)

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
        pred: int,
        y: int,
        alpha: float,
    ) -> _TTrial: ...

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
        if pA < 0.5:  # noqa: PLR2004
            return Certificate(_ABSTAIN, 0.0)
        radius = sigma * cast("float", norm.ppf(pA).item())
        return Certificate(y, radius, pA=pA)
