from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.exceptions import TorchMetricsUserError

from benchmark_ctrs.models.smooth import SmoothedClassifier

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from typing_extensions import Literal

    from benchmark_ctrs.models.smooth import HParams


class CertifiedRadius(Metric):
    higher_is_better = True
    is_differentiable = False
    plot_lower_bound = 0.0

    indices: list[Tensor]
    radii: list[Tensor]
    smooth: SmoothedClassifier

    feature_network = "smooth"

    def __init__(
        self,
        base_classifier: Module,
        params: HParams,
        *,
        num_classes: int,
        sigma: float,
        start: int = 0,
        skip: int = 1,
        max_: int | None = None,
        reduction: Literal["mean", "max", "min", "none"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if reduction not in {"mean", "min", "max", "none"}:
            raise TorchMetricsUserError(f"Unknown reduction mode: '{reduction}'")

        self.reduction = reduction
        self.start = start
        self.skip = skip
        self.max = max_
        self.smooth = SmoothedClassifier(
            base_classifier,
            num_classes=num_classes,
            sigma=sigma,
            params=params,
        )
        self.add_state("radii", default=[], dist_reduce_fx="cat")
        if self.reduction == "none":
            self.add_state("indices", default=[], dist_reduce_fx="cat")

    def update(self, inputs: Tensor):
        indices = torch.arange(
            start=self.start,
            end=self.max if self.max is not None else inputs.size(0),
            step=self.skip,
        )
        if self.reduction == "none":
            self.indices.append(indices)

        radii = np.fromiter(
            iter=(
                self.smooth(x, certify=True).certified_radius for x in inputs[indices]
            ),
            dtype=float,
        )
        self.radii.append(torch.as_tensor(radii, device=self.device))

    def compute(self) -> Tensor | tuple[Tensor, Tensor]:
        radii = dim_zero_cat(self.radii)
        if self.reduction == "max":
            return radii.max()
        if self.reduction == "min":
            return radii.min()
        if self.reduction == "none":
            indices = dim_zero_cat(self.indices)
            return indices, radii
        return radii.mean()
