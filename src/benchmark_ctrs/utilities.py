from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor
from typing_extensions import TypeIs

from benchmark_ctrs.modules.module import BaseModule

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from benchmark_ctrs.types import StepOutput


def check_valid_submodule(
    pl_module: L.LightningModule,
) -> TypeIs[BaseModule]:
    return isinstance(pl_module, BaseModule)


def check_valid_step_output(value: STEP_OUTPUT) -> TypeIs[StepOutput]:
    return (
        isinstance(value, dict)
        and ("loss" not in value or isinstance(value["loss"], Tensor))
        and ("prediction" not in value or isinstance(value["predictions"], Tensor))
    )
