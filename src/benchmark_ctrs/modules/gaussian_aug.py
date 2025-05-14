from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from benchmark_ctrs.modules.rs_training import RSTrainingModule, StepOutput

if TYPE_CHECKING:
    from benchmark_ctrs.modules.rs_training import Batch


class GaussianAug(RSTrainingModule):
    @override
    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutput:
        return self._default_eval_step(batch)
