from __future__ import annotations

from typing import Any, Final

from torch.utils.data import random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.classification_module import (
    ClassificationDataModule,
    Datasets,
)
from benchmark_ctrs.models import Architectures


class MNIST(ClassificationDataModule):
    __means: Final = [0.0]
    __sds: Final = [1.0]

    def __init__(self, batch_size: int = 400, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_size, *args, **kwargs)

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
