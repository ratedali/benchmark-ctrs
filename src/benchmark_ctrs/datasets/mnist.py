from typing import Any, ClassVar, Final

from torch.utils.data import random_split
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.models import Architecture

__all__ = ["MNIST"]


class MNIST(BaseDataModule):
    __means: Final = [0.0]
    __sds: Final = [1.0]

    default_arch: ClassVar = Architecture.LeNet

    def __init__(self, *args: Any, batch_size: int = 256, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, batch_size=batch_size)
        self.name = "mnist"

    def prepare_data(self) -> None:
        mnist.MNIST(self.cache_dir, train=True, download=True)
        mnist.MNIST(self.cache_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = mnist.MNIST(self.cache_dir, train=True, transform=ToTensor())
            if self.validation > 0:
                total = len(dataset) if isinstance(self.validation, int) else 1.0
                self._train, self._val = random_split(
                    dataset=dataset,
                    lengths=(total - self.validation, self.validation),
                )
            else:
                self._train = dataset
        else:
            dataset = mnist.MNIST(self.cache_dir, train=False, transform=ToTensor())
            if stage == "test":
                self._test = dataset
            elif stage == "predict":
                self._predict = dataset

    @property
    @override
    def classes(self) -> int:
        return 10

    @property
    @override
    def mean(self) -> list[float]:
        return MNIST.__means

    @property
    @override
    def std(self) -> list[float]:
        return MNIST.__sds
