from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler, SequentialLR, StepLR
from typing_extensions import TypeIs

from benchmark_ctrs.types import Batch, StepOutput


def GradualStepLR(
    optimizer: Optimizer,
    warmup_factor: float = 0.1,
    warmup_iters: int = 1,
    step_size: int = 50,
    gamma: float = 0.1,
) -> LRScheduler:
    warmup_phase = ConstantLR(optimizer, factor=warmup_factor, total_iters=warmup_iters)
    normal_phase = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return SequentialLR(
        optimizer,
        [warmup_phase, normal_phase],
        milestones=[warmup_iters],
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
