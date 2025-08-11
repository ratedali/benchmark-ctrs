from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule

if TYPE_CHECKING:
    from benchmark_ctrs.types import Batch, StepOutput


@dataclass(frozen=True)
class HParams(BaseHParams):
    learning_rate: float = 0.1
    lr_decay: float = 0.1
    lr_step: int = 60
    momentum: float = 0.9
    weight_decay: float = 1e-4


class GaussianAug(BaseModule):
    def __init__(self, *args, params: HParams, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)

    @override
    def training_step(
        self,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> StepOutput:
        return self._default_eval_step(batch, add_noise=True)
