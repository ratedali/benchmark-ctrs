from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from benchmark_ctrs.types import Batch, StepOutput


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
