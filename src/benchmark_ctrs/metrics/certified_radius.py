from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.exceptions import TorchMetricsUserError

from benchmark_ctrs.models import smooth

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from typing_extensions import Literal


@dataclasses.dataclass(frozen=True)
class Params(smooth.HParams):
    start: int = 0
    skip: int = 1
    max_: int | None = None


class CertifiedRadius(Metric):
    higher_is_better = True
    is_differentiable = False
    plot_lower_bound = 0.0
    feature_network: ClassVar = "_smooth"

    _indices: list[Tensor]
    _predictions: list[Tensor]
    _radii: list[Tensor]
    _smooth: smooth.SmoothedClassifier

    def __init__(
        self,
        base_classifier: Module,
        params: Params,
        *,
        num_classes: int,
        sigma: float,
        reduction: Literal["mean", "max", "min", "none"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if reduction not in {"mean", "min", "max", "none"}:
            raise TorchMetricsUserError(f"Unknown reduction mode: '{reduction}'")

        self._reduction = reduction
        self._start = params.start
        self._skip = params.skip
        self._max = params.max_
        self._smooth = smooth.SmoothedClassifier(
            base_classifier,
            num_classes=num_classes,
            sigma=sigma,
            params=params,
        )
        self.add_state("_radii", default=[], dist_reduce_fx="cat")
        if self._reduction == "none":
            self.add_state("_predictions", default=[], dist_reduce_fx="cat")
            self.add_state("_indices", default=[], dist_reduce_fx="cat")

    def update(self, inputs: Tensor):
        indices = torch.arange(
            start=self._start,
            end=self._max if self._max is not None else inputs.size(0),
            step=self._skip,
        )
        certs = [self._smooth.forward(x, certify=True) for x in inputs[indices]]
        radii = torch.tensor(
            [cert.certified_radius for cert in certs],
            dtype=torch.float,
            device=self.device,
        )
        self._radii.append(torch.as_tensor(radii, device=self.device))

        if self._reduction == "none":
            predictions = torch.tensor(
                [
                    cert.prediction if isinstance(cert.prediction, int) else -1
                    for cert in certs
                ],
                dtype=torch.int,
                device=self.device,
            )
            self._indices.append(indices)
            self._predictions.append(predictions)

    def compute(self) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        radii = dim_zero_cat(self._radii)
        if self._reduction == "none":
            indices = dim_zero_cat(self._indices)
            predictions = dim_zero_cat(self._predictions)
            return indices, predictions, radii
        if radii.numel() == 0:
            return torch.tensor([0.0]).to(radii)
        if self._reduction == "max":
            return radii.max()
        if self._reduction == "min":
            return radii.min()
        return radii.mean()
