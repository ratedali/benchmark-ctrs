from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark_ctrs.model import Architecture, ModelWrapper


class TrainingCheckpoint:
    """
    This class is used to save the model and optimizer state at a given epoch.
    """

    DEFAULT_FILE_NAME: Final = "checkpoint.pth.tar"

    def __init__(
        self,
        epoch: int,
        arch: Architecture,
        state_dict: dict,
        optimizer: dict,
    ):
        self.epoch = epoch
        self.arch: Architecture = arch
        self.state_dict = state_dict
        self.optimizer = optimizer

    def save(self, path: Path):
        """Saves the checkpoint to a file

        Args:
            path (pathlib.Path): the checkpoint file path
            (if directory, file name will be 'TrainingCheckpoint.DEFAULT_FILE_NAME')
        """
        if path.is_dir():
            path = path / self.DEFAULT_FILE_NAME

        torch.save(
            {
                "epoch": self.epoch,
                "arch": self.arch,
                "state_dict": self.state_dict,
                "optimizer": self.optimizer,
            },
            path,
        )

    @staticmethod
    def capture(
        model_wrapper: ModelWrapper,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> TrainingCheckpoint:
        """Factory method
        Creates a training checkpoint for a given model and optimizer

        Args:
            model (ModelWrapper): the model under training
            optimizer (torch.optim.Optimizer): the optimizer used for training
            epoch (int): the current epoch of training
        Returns:
            TrainingCheckpoint: the training checkpoint capturing
            the current states of the arguments
        """
        return TrainingCheckpoint(
            epoch=epoch,
            arch=model_wrapper.arch,
            state_dict=deepcopy(model_wrapper.model.state_dict()),
            optimizer=deepcopy(optimizer.state_dict()),
        )

    @classmethod
    def load(cls, path: Path) -> TrainingCheckpoint:
        """loads a previously saved training checkpoint.

        Args:
            path (pathlib.Path): the checkpoint file path
            (if directory, file name will be 'TrainingCheckpoint.DEFAULT_FILE_NAME')
        Returns:
            TrainingCheckpoint: the loaded training checkpoint
        """
        if path.is_dir():
            path = path / cls.DEFAULT_FILE_NAME

        checkpoint_dict = torch.load(path, lambda storage, _loaction: storage)
        return TrainingCheckpoint(
            epoch=checkpoint_dict["epoch"],
            arch=checkpoint_dict["arch"],
            state_dict=checkpoint_dict["state_dict"],
            optimizer=checkpoint_dict["optimizer"],
        )
