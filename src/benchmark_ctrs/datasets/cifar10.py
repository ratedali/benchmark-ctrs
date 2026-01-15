from typing import Any, ClassVar, Final

from torch.utils.data import random_split
from torchvision.datasets import cifar
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from typing_extensions import override

from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.models import Architecture


class CIFAR10(BaseDataModule):
    __means: Final = [0.4914, 0.4822, 0.4465]
    __sds: Final = [0.2470, 0.2435, 0.2616]

    default_arch: ClassVar = Architecture.CIFARResNet110

    def __init__(self, *args: Any, batch_size: int = 256, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, batch_size=batch_size)
        self.name = "cifar10"

    def prepare_data(self) -> None:
        cifar.CIFAR10(self.cache_dir, train=True, download=True)
        cifar.CIFAR10(self.cache_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = cifar.CIFAR10(
                self.cache_dir,
                train=True,
                transform=Compose(
                    [
                        RandomCrop(32, padding=4),
                        RandomHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
            if self.validation > 0:
                total = len(dataset) if isinstance(self.validation, int) else 1.0
                self._train, self._val = random_split(
                    dataset=dataset,
                    lengths=(total - self.validation, self.validation),
                )
        else:
            dataset = cifar.CIFAR10(
                self.cache_dir,
                train=False,
                transform=ToTensor(),
            )
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
        return CIFAR10.__means

    @property
    @override
    def std(self) -> list[float]:
        return CIFAR10.__sds
