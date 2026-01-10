import dataclasses
from typing import Any, ClassVar, Literal, NamedTuple, Union, cast

import numpy as np
import torch
from lightning.pytorch.utilities import LightningEnum
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.exceptions import TorchMetricsUserError

from benchmark_ctrs.certification import CertificationMethod, is_abstain


class _Reduction(LightningEnum):
    Mean = "mean"
    Max = "max"
    Min = "min"
    None_ = "none"


@dataclasses.dataclass(frozen=True)
class Params:
    sigma: float
    alpha: float = 0.001
    start: int = 0
    skip: int = 1
    max_: Union[int, Literal[False]] = False


class CertificationResult(NamedTuple):
    indices: Tensor
    predictions: Tensor
    radii: Tensor


class CertifiedRadius(Metric):
    higher_is_better = True
    is_differentiable = False
    plot_lower_bound = 0.0
    feature_network: ClassVar = "_model"

    _radii: Union[Tensor, list[Tensor]]
    _indices: list[Tensor]
    _predictions: list[Tensor]
    _total: Tensor
    _model: torch.nn.Module
    _certifier: CertificationMethod

    def __init__(
        self,
        model: torch.nn.Module,
        certifier: CertificationMethod,
        params: Params,
        *,
        num_classes: int,
        reduction: Literal["mean", "max", "min", "none"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if reduction not in list(_Reduction):
            raise TorchMetricsUserError(
                f"Unknown reduction mode '{reduction}'. "
                f"Supported values: {[x.value for x in _Reduction]}"
            )
        self._num_classes = num_classes
        self._sigma = params.sigma
        self._alpha = params.alpha
        self._start = params.start
        self._skip = params.skip
        self._max: Union[int, Literal[False]] = params.max_
        self._model = model
        self._certifier = certifier
        self._reduction = reduction

        if self._reduction == "none":
            self.add_state("_radii", default=[], dist_reduce_fx="cat")
            self.add_state("_predictions", default=[], dist_reduce_fx="cat")
            self.add_state("_indices", default=[], dist_reduce_fx="cat")
        elif self._reduction == "max":
            self.add_state("_radii", default=torch.tensor(0), dist_reduce_fx="max")
        elif self._reduction == "min":
            self.add_state(
                "_radii",
                default=torch.tensor(torch.inf, dtype=torch.float),
                dist_reduce_fx="min",
            )
        elif self._reduction == "mean":
            self.add_state(
                "_radii",
                default=torch.tensor(0.0, dtype=torch.float),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "_total",
                default=torch.tensor(0, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    @torch.inference_mode()
    def update(self, inputs: Tensor, targets: Tensor) -> None:
        indices = torch.arange(
            start=self._start,
            end=self._max if self._max is not False else inputs.size(0),
            step=self._skip,
            device=self.device,
            dtype=torch.long,
        )

        certs = self._certifier.certify_batch(
            self._model,
            data=(inputs, targets),
            sigma=self._sigma,
            alpha=self._alpha,
            num_classes=self._num_classes,
        )
        radii = torch.tensor([cert.radius for cert in certs], device=self.device)

        predictions = torch.tensor(
            [
                cert.prediction if not is_abstain(cert.prediction) else -1
                for cert in certs
            ],
            dtype=torch.long,
            device=self.device,
        )

        if self._reduction == "none":
            cast("list[Tensor]", self._radii).append(radii)
            self._indices.append(indices)
            self._predictions.append(predictions)
        else:
            _radii = cast("Tensor", self._radii)
            correct = targets[indices] == predictions

            if self._reduction == "mean":
                self._radii += radii[correct].sum()
                self._total += radii.numel()
            elif self._reduction == "max" and correct.any().item():
                self._radii = torch.max(_radii, radii[correct].max())
            elif self._reduction == "min" and correct.any().item():
                self._radii = torch.min(_radii, radii[correct].min())

    @torch.inference_mode()
    def compute(self) -> Union[Tensor, CertificationResult]:
        """Compute the certified radius metric value

        Returns:
            (Tensor | CertificationResult): returns the CertificationResults tuple
            if reduction="none", otherwise it returns the aggregated certified radii
        """
        if self._reduction == "none":
            radii = dim_zero_cat(self._radii)
            indices = dim_zero_cat(self._indices)
            predictions = dim_zero_cat(self._predictions)
            return CertificationResult(
                indices=indices,
                predictions=predictions,
                radii=radii,
            )
        radii = cast("Tensor", self._radii)
        if self._reduction == "max":
            return radii
        if self._reduction == "min":
            return (
                radii
                if not torch.isinf(radii).any().item()
                else torch.tensor(0.0, dtype=torch.float, device=self.device)
            )

        return radii.float() / self._total
