from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import lightning as L
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

if TYPE_CHECKING:
    from typing import Any

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
        with_ids: bool = False,
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
        if (dataset := self._train) is None:
            raise ValueError("Training split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if (dataset := self._val) is None:
            raise ValueError("Validation split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            pin_memory=True,
            persistent_workers=True,
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if (dataset := self._test) is None:
            raise ValueError("Testing split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            pin_memory=True,
            persistent_workers=True,
        )

    @override
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        if (dataset := self._predict) is None:
            raise ValueError("Testing split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["workers"],
            pin_memory=True,
            persistent_workers=True,
        )


class _WithIdDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.__dataset = dataset

    def __getitem__(self, index) -> Any:
        item = self.__dataset[index]
        return (*item, index)

    def __len__(self) -> int:
        if not isinstance(self.__dataset, Sized):
            raise NotImplementedError(
                f"{self.__dataset.__class__.__name__}does not implement __len__"
            )
        return len(self.__dataset)
