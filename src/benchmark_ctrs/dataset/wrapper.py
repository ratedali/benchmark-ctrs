import pathlib

import torch
from torchvision import datasets, transforms
from typing_extensions import Literal, TypeAlias

Dataset: TypeAlias = Literal["mnist", "cifar10", "cifar100", "imagenet"]
Split: TypeAlias = Literal["train", "test"]


class DatasetWrapper:
    def __init__(self, dataset: Dataset, data_dir: pathlib.Path):
        self.dataset: Dataset = dataset
        self._data_dir = data_dir

        if dataset == "mnist":
            self._classes = 10
            self._mean = [0.0]
            self._sd = [1.0]
        elif dataset == "cifar10":
            self._classes = 10
            self._mean = [0.4914, 0.4822, 0.4465]
            self._sd = [0.2023, 0.1994, 0.2010]
        elif dataset == "cifar100":
            self._classes = 100
            self._mean = [0.5071, 0.4867, 0.4408]
            self._sd = [0.2675, 0.2565, 0.2761]
        elif dataset == "imagenet":
            self._classes = 1000
            self._mean = [0.485, 0.456, 0.406]
            self._sd = [0.229, 0.224, 0.225]
        else:
            raise ValueError("Invalid dataset")

    @property
    def classes(self):
        return self._classes

    @property
    def mean(self) -> list[float]:
        return self._mean

    @property
    def sd(self) -> list[float]:
        return self._sd

    def get_transforms(self, split: Split):
        if self.dataset in {"cifar10", "cifar100"} and split == "train":
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ],
            )

        if self.dataset == "imagenet":
            if split == "train":
                return transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ],
                )

            if split == "test":
                return transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ],
                )

        return transforms.ToTensor()

    def get_split(self, split: Split) -> torch.utils.data.Dataset[torch.Tensor]:
        if self.dataset == "mnist":
            return datasets.MNIST(
                root=self._data_dir,
                train=(split == "train"),
                download=True,
                transform=self.get_transforms(split),
            )
        if self.dataset == "cifar10":
            return datasets.CIFAR10(
                root=self._data_dir,
                train=(split == "train"),
                download=True,
                transform=self.get_transforms(split),
            )
        if self.dataset == "cifar100":
            return datasets.CIFAR100(
                root=self._data_dir,
                train=(split == "train"),
                download=True,
                transform=self.get_transforms(split),
            )
        if self.dataset == "imagenet":
            return datasets.ImageNet(
                root=self._data_dir,
                split="train" if split == "train" else "val",
                transform=self.get_transforms(split),
            )
        raise ValueError("Invalid dataset name")
