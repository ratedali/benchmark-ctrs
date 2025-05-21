from __future__ import annotations

from typing import Any, Final

from torch.utils.data import random_split
from torchvision.datasets import imagenet
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from typing_extensions import override

from benchmark_ctrs.datasets.classification_module import (
    ClassificationDataModule,
    Datasets,
)
from benchmark_ctrs.models import Architectures


class ImageNet(ClassificationDataModule):
    __means: Final = [0.485, 0.456, 0.406]
    __sds: Final = [0.229, 0.224, 0.225]

    def __init__(self, *args: Any, batch_size: int = 64, **kwargs: Any):
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.__train_transforms = Compose(
            [
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
            ],
        )
        self.__test_transforms = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ],
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train, self._val = random_split(
                dataset=imagenet.ImageNet(
                    self._cache_dir,
                    split="train",
                    transform=self.__train_transforms,
                ),
                lengths=(0.8, 0.2),
            )
        elif stage == "test":
            self._test = imagenet.ImageNet(
                self._cache_dir, split="val", transform=self.__test_transforms
            )
        elif stage == "predict":
            self._predict = imagenet.ImageNet(
                self._cache_dir, split="val", transform=self.__test_transforms
            )

    @property
    @override
    def default_arch(self) -> Architectures:
        return Architectures.Resnet50

    @property
    @override
    def dataset(self) -> Datasets:
        return Datasets.CIFAR_10

    @property
    @override
    def classes(self) -> int:
        return 10

    @property
    @override
    def means(self) -> list[float]:
        return ImageNet.__means

    @property
    @override
    def sds(self) -> list[float]:
        return ImageNet.__sds
