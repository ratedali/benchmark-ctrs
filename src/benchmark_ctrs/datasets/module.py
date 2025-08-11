from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import lightning as L
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

    from benchmark_ctrs.models import Architecture


class BaseDataModule(L.LightningDataModule, ABC):
    __default_cache_dir: ClassVar = Path("datasets_cache")

    def __init__(
        self,
        *,
        batch_size: int,
        workers: int = 4,
        cache_dir: Path | None = None,
    ):
        super().__init__()
        self._cache_dir = cache_dir or BaseDataModule.__default_cache_dir
        self.save_hyperparameters(ignore="cache_dir")

        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None
        self._predict: Dataset | None = None

    @property
    def default_arch(self) -> Architecture | None:
        return None

    @property
    @abstractmethod
    def classes(self) -> int: ...

    @property
    @abstractmethod
    def means(self) -> list[float]: ...

    @property
    @abstractmethod
    def sds(self) -> list[float]: ...

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self._train is None:
            raise ValueError("Training split not initialized")
        return DataLoader(
            self._train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            persistent_workers=True,
            shuffle=True,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val is None:
            raise ValueError("Validation split not initialized")
        return DataLoader(
            self._val,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            persistent_workers=True,
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test is None:
            raise ValueError("Testing split not initialized")
        return DataLoader(
            self._test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            persistent_workers=True,
        )

    @override
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        if self._predict is None:
            raise ValueError("Testing split not initialized")
        return DataLoader(
            self._predict,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            persistent_workers=True,
        )
