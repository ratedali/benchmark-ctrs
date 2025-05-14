from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from torch.utils.data import random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.classification_module import (
    ClassificationDataModule,
    DataModuleParams,
    Datasets,
)
from benchmark_ctrs.models import Architectures


@dataclass(frozen=True)
class MNISTParams(DataModuleParams):
    batch_size: int = 400


_default_params = MNISTParams()


class MNIST(ClassificationDataModule[MNISTParams]):
    __means: Final = [0.0]
    __sds: Final = [1.0]

    def __init__(self, params: MNISTParams = _default_params) -> None:
        super().__init__(params)

    def prepare_data(self) -> None:
        mnist.MNIST(self.params.cache_dir, train=True, download=True)
        mnist.MNIST(self.params.cache_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train, self._val = random_split(
                dataset=mnist.MNIST(
                    self.params.cache_dir, train=True, transform=ToTensor()
                ),
                lengths=(55000, 5000),
            )
        elif stage == "test":
            self._test = mnist.MNIST(
                self.params.cache_dir, train=False, transform=ToTensor()
            )
        elif stage == "predict":
            self._predict = mnist.MNIST(
                self.params.cache_dir, train=False, transform=ToTensor()
            )

    @property
    @override
    def default_arch(self) -> Architectures:
        return Architectures.LeNet

    @property
    @override
    def dataset(self) -> Datasets:
        return Datasets.MNIST

    @property
    @override
    def classes(self) -> int:
        return 10

    @property
    @override
    def means(self) -> list[float]:
        return MNIST.__means

    @property
    @override
    def sds(self) -> list[float]:
        return MNIST.__sds
