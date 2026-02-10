from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer, lr_scheduler
from typing_extensions import TypeIs

from benchmark_ctrs.types import Batch, LRSchedulerCallable, StepOutput

__all__ = [
    "ChainedLR",
    "GradualStepLR",
    "SequentialLR",
    "check_valid_step_output",
    "generate_repeats",
]


class SequentialLR(lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRSchedulerCallable],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            schedulers=[scheduler(optimizer) for scheduler in schedulers],
            milestones=milestones,
            last_epoch=last_epoch,
        )


class GradualStepLR(lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_factor: float = 0.1,
        warmup_iters: int = 1,
        step_size: int = 50,
        gamma: float = 0.1,
    ) -> None:
        warmup_phase = lr_scheduler.ConstantLR(
            optimizer,
            factor=warmup_factor,
            total_iters=warmup_iters,
        )
        normal_phase = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        super().__init__(
            optimizer,
            [warmup_phase, normal_phase],
            milestones=[warmup_iters],
        )


class ChainedLR(lr_scheduler.ChainedScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRSchedulerCallable],
    ) -> None:
        super().__init__(
            schedulers=[scheduler(optimizer) for scheduler in schedulers],
            optimizer=optimizer,
        )


def check_valid_step_output(value: STEP_OUTPUT) -> TypeIs[StepOutput]:
    return (
        isinstance(value, dict)
        and ("loss" not in value or isinstance(value["loss"], Tensor))
        and ("prediction" not in value or isinstance(value["predictions"], Tensor))
    )


def generate_repeats(batch: Batch, k: int) -> Batch:
    inputs, *labels = batch

    batch_size, *rest = inputs.shape
    repeats_shape = [batch_size * k, *rest]

    repeats = [1 for _ in range(inputs.dim())]
    input_repeats = inputs.repeat(repeats).view(repeats_shape)

    label_repeats = [label.unsqueeze(1).expand(-1, k).flatten() for label in labels]

    return (input_repeats, *label_repeats)
