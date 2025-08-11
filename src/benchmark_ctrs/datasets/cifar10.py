from __future__ import annotations

from typing import Any, Final

from torch.utils.data import random_split
from torchvision.datasets import cifar
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.models import Architecture


class CIFAR10(BaseDataModule):
    __means: Final = [0.4914, 0.4822, 0.4465]
    __sds: Final = [0.2470, 0.2435, 0.2616]

    def __init__(self, *args: Any, batch_size: int = 400, **kwargs: Any):
        super().__init__(*args, **kwargs, batch_size=batch_size)
        self.name = "cifar10"

    def prepare_data(self) -> None:
        cifar.CIFAR10(self._cache_dir, train=True, download=True)
        cifar.CIFAR10(self._cache_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train, self._val = random_split(
                dataset=cifar.CIFAR10(
                    self._cache_dir,
                    train=True,
                    transform=Compose(
                        [
                            RandomCrop(32, padding=4),
                            RandomHorizontalFlip(),
                            ToTensor(),
                        ]
                    ),
                ),
                lengths=(49000, 1000),
            )
        elif stage == "test":
            self._test = cifar.CIFAR10(
                self._cache_dir, train=False, transform=ToTensor()
            )
        elif stage == "predict":
            self._predict = cifar.CIFAR10(
                self._cache_dir, train=False, transform=ToTensor()
            )

    @property
    @override
    def default_arch(self) -> Architecture:
        return Architecture.Resnet110

    @property
    @override
    def classes(self) -> int:
        return 10

    @property
    @override
    def means(self) -> list[float]:
        return CIFAR10.__means

    @property
    @override
    def sds(self) -> list[float]:
        return CIFAR10.__sds
