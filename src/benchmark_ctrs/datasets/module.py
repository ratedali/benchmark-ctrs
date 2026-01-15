import os
from abc import ABC, abstractmethod
from collections.abc import Sized
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import lightning as L
from lightning.pytorch.utilities import suggested_max_num_workers
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from benchmark_ctrs.models import Architecture


class BaseDataModule(L.LightningDataModule, ABC):
    __default_cache_dir: ClassVar = Path("datasets_cache")

    default_arch: ClassVar[Optional[Architecture]] = None

    def __init__(
        self,
        *,
        batch_size: int,
        validation: Union[int, float],
        workers: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        with_ids: bool = False,
        shuffle_train: bool = True,
    ):
        super().__init__()
        self.cache_dir = cache_dir or os.environ.get(
            "DATASETS_CACHE",
            BaseDataModule.__default_cache_dir,
        )

        self.workers = workers or suggested_max_num_workers(
            self.trainer.world_size // self.trainer.num_nodes if self.trainer else 1
        )

        if validation <= 0:
            raise ValueError("validation must be a non-negative number.")
        if isinstance(validation, float) and validation > 1:
            raise ValueError("validation cannot be a float greater than 1.")

        self.validation = validation

        self.save_hyperparameters(ignore="cache_dir")

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None
        self._predict: Optional[Dataset] = None

    @property
    @abstractmethod
    def classes(self) -> int: ...

    @property
    @abstractmethod
    def mean(self) -> list[float]: ...

    @property
    @abstractmethod
    def std(self) -> list[float]: ...

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if (dataset := self._train) is None:
            raise RuntimeError("Training split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=self.hparams["shuffle_train"],
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if (dataset := self._val) is None:
            raise RuntimeError("Validation split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=True,
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if (dataset := self._test) is None:
            raise RuntimeError("Testing split not initialized")
        if self.hparams["with_ids"]:
            dataset = _WithIdDatasetWrapper(dataset)

        return DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.workers,
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
            num_workers=self.workers,
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
