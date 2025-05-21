from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Generic

import lightning as L
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypeVar, override

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

    from benchmark_ctrs.models import Architectures


class Datasets(Enum):
    MNIST = "MNIST"
    CIFAR_10 = "CIFAR-10"
    ImageNet = "ImageNet"


@dataclass(frozen=True)
class DataModuleParams:
    cache_dir: Path = field(default=Path("datasets_cache"))
    workers: int = 4


_Tparams = TypeVar("_Tparams", bound=DataModuleParams, default=DataModuleParams)


class ClassificationDataModule(L.LightningDataModule, ABC, Generic[_Tparams]):
    def __init__(
        self,
        batch_size: int,
        params: _Tparams,
    ):
        super().__init__()

        self._batch_size = batch_size
        self.__params = params

        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None
        self._predict: Dataset | None = None

    @property
    def params(self) -> _Tparams:
        return self.__params

    @property
    def default_arch(self) -> Architectures | None:
        return None

    @property
    @abstractmethod
    def dataset(self) -> Datasets: ...

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
            batch_size=self._batch_size,
            num_workers=self.__params.workers,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self._val is None:
            raise ValueError("Validation split not initialized")
        return DataLoader(
            self._val,
            batch_size=self._batch_size,
            num_workers=self.__params.workers,
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self._test is None:
            raise ValueError("Testing split not initialized")
        return DataLoader(
            self._test,
            batch_size=self._batch_size,
            num_workers=self.__params.workers,
        )

    @override
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        if self._predict is None:
            raise ValueError("Testing split not initialized")
        return DataLoader(
            self._predict,
            batch_size=self._batch_size,
            num_workers=self.__params.workers,
        )
