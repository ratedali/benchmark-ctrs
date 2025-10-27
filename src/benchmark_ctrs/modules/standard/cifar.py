from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule

if TYPE_CHECKING:
    from benchmark_ctrs.types import Batch, StepOutput


@dataclasses.dataclass(frozen=True)
class HParams:
    learning_rate: float = 0.1
    lr_decay: float = 0.1
    lr_step: int = 30
    momentum: float = 0.9
    weight_decay: float = 1e-4


class CIFARStandard(BaseModule):
    def __init__(self, *args, params: HParams, **kwargs) -> None:
        super().__init__(
            *args,
            params=BaseHParams(sigma=0.0, **dataclasses.asdict(params)),
            **kwargs,
        )

    @override
    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def _default_eval_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        inputs, targets = batch
        predictions = self.forward(inputs, add_noise=False)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
