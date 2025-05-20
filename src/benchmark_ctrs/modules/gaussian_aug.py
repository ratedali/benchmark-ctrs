from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import override

from benchmark_ctrs.models import Architectures
from benchmark_ctrs.models.smooth import HParams
from benchmark_ctrs.modules.rs_training import HParams, RandomizedSmoothing

if TYPE_CHECKING:
    from benchmark_ctrs.modules.rs_training import Batch, StepOutput


@dataclass(frozen=True)
class GaussianAugHParams(HParams):
    learning_rate: float = 0.1


class GaussianAug(RandomizedSmoothing):
    def __init__(self, *args, params: GaussianAugHParams, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)

    @override
    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._default_eval_step(batch)
