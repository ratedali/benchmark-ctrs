from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark_ctrs.dataset import Dataset
    from benchmark_ctrs.model import Architecture


class TrainingParameters(TypedDict):
    dataset: Dataset
    data_dir: Path
    architecture: Architecture
    id: int
    noise_sd: float
    epochs: int
    batch_size: int
    num_workers: int
    optimizer: Literal["sgd"]
    loss: Literal["cross-entropy"]
    validation: Literal["kfold", "set"]
    validation_set_split: float
    validation_kfold: int
    lr: float
    lr_schedule: Literal["step", "constant"]
    lr_schedule_gamma: float
    lr_step_size: int
    weight_decay: float
    momentum: float
    save: bool
    rundir: Path
    resume: bool
    resume_path: Path | None
    log_freq: int
    log_grads: bool
