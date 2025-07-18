from __future__ import annotations

from typing import Any, Final

from torch.utils.data import random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.models import Architecture


class MNIST(BaseDataModule):
    __means: Final = [0.0]
    __sds: Final = [1.0]

    def __init__(self, *args: Any, batch_size: int = 400, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, batch_size=batch_size)
        self.name = "mnist"

    def prepare_data(self) -> None:
        mnist.MNIST(self._cache_dir, train=True, download=True)
        mnist.MNIST(self._cache_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train, self._val = random_split(
                dataset=mnist.MNIST(self._cache_dir, train=True, transform=ToTensor()),
                lengths=(55000, 5000),
            )
        elif stage == "test":
            self._test = mnist.MNIST(self._cache_dir, train=False, transform=ToTensor())
        elif stage == "predict":
            self._predict = mnist.MNIST(
                self._cache_dir, train=False, transform=ToTensor()
            )

    @property
    @override
    def default_arch(self) -> Architecture:
        return Architecture.LeNet

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
