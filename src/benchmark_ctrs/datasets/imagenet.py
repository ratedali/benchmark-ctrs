from typing import Any, ClassVar, Final

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

from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.models import Architecture


class ImageNet(BaseDataModule):
    __means: Final = [0.485, 0.456, 0.406]
    __sds: Final = [0.229, 0.224, 0.225]

    default_arch: ClassVar = Architecture.ResNet50

    def __init__(self, *args: Any, batch_size: int = 64, **kwargs: Any) -> None:
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.name = "imagenet"
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
                lengths=(1280167, 1000),
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
    def classes(self) -> int:
        return 1000

    @property
    @override
    def mean(self) -> list[float]:
        return ImageNet.__means

    @property
    @override
    def std(self) -> list[float]:
        return ImageNet.__sds
