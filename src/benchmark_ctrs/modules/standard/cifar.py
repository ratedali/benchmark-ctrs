from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR, MultiStepLR, SequentialLR
from typing_extensions import override

from benchmark_ctrs.models import Architecture
from benchmark_ctrs.modules import BaseHParams, BaseModule

if TYPE_CHECKING:
    from benchmark_ctrs.types import CONFIGURE_OPTIMIZERS, Batch, StepOutput


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

    def configure_optimizers(self) -> CONFIGURE_OPTIMIZERS:
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            momentum=self.hparams["momentum"],
            weight_decay=self.hparams["weight_decay"],
        )
        step = self.hparams["lr_step"]
        milestones = list(range(step, self.trainer.max_epochs or 150, step))
        step_lr = MultiStepLR(optimizer, milestones, gamma=0.1)
        if self._arch == Architecture.CIFARResNet110:
            warmup_lr = ConstantLR(optimizer, factor=0.1, total_iters=1)
            scheduler = SequentialLR(optimizer, [warmup_lr, step_lr], milestones=[1])
        else:
            scheduler = step_lr
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

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
        loss = self._criterion(predictions, targets)
        return {"loss": loss, "predictions": predictions}
