from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim import SGD
from typing_extensions import override

from benchmark_ctrs.modules import BaseHParams, BaseModule

if TYPE_CHECKING:
    from benchmark_ctrs.types import CONFIGURE_OPTIMIZERS, Batch, StepOutput


class MNISTStandard(BaseModule):
    def __init__(self, *args, learning_rate: float = 0.1, **kwargs) -> None:
        super().__init__(
            *args,
            params=BaseHParams(
                sigma=0.0,
                learning_rate=learning_rate,
                lr_decay=1,
                lr_step=-1,
                momentum=0,
                weight_decay=0,
            ),
            **kwargs,
        )

    @override
    def configure_optimizers(self) -> CONFIGURE_OPTIMIZERS:
        return SGD(self.parameters(), lr=self.hparams["learning_rate"])

    @override
    def training_step(self, batch: Batch, *args, **kwargs) -> StepOutput:
        return self._default_eval_step(batch)

    @override
    def _default_eval_step(self, batch: Batch, *args, **kwargs) -> StepOutput:
        inputs, targets = batch[:2]
        predictions = self.forward(inputs, add_noise=False)
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
