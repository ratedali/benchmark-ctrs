from abc import ABC, abstractmethod
from typing import NamedTuple

from torch import Tensor
from typing_extensions import TypeIs

from benchmark_ctrs.types import Batch, Classifier

__all__ = [
    "_ABSTAIN",
    "Certificate",
    "CertificationMethod",
    "Prediction",
    "is_abstain",
]


class _ABSTAIN_TYPE: ...


_ABSTAIN = _ABSTAIN_TYPE()

Prediction = int | _ABSTAIN_TYPE


def is_abstain(prediction: Prediction) -> TypeIs[_ABSTAIN_TYPE]:
    return prediction == _ABSTAIN


class Certificate(NamedTuple):
    prediction: Prediction
    radius: float
    pA: float | None = None
    pB: float | None = None


class CertificationMethod(ABC):
    @abstractmethod
    def certify(
        self,
        model: Classifier,
        x: Tensor,
        y: int,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> Certificate: ...

    def certify_batch(
        self,
        model: Classifier,
        data: Batch,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> list[Certificate]:
        inputs, labels = data
        return [
            self.certify(
                model,
                x=inputs[i, ...],
                y=int(labels[i].item()),
                sigma=sigma,
                alpha=alpha,
                num_classes=num_classes,
            )
            for i in range(inputs.size(0))
        ]

    @abstractmethod
    def predict(
        self,
        model: Classifier,
        x: Tensor,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> Certificate: ...

    def predict_batch(
        self,
        model: Classifier,
        inputs: Tensor,
        *,
        sigma: float,
        alpha: float = 0.001,
        num_classes: int,
    ) -> list[Certificate]:
        return [
            self.predict(
                model,
                x=inputs[i, ...],
                sigma=sigma,
                alpha=alpha,
                num_classes=num_classes,
            )
            for i in range(inputs.size(0))
        ]
