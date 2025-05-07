from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchvision import models
from typing_extensions import Literal, Self, TypeAlias

from benchmark_ctrs.model.architectures import cifar_resnet, lenet
from benchmark_ctrs.model.normalization import Normalization

if TYPE_CHECKING:
    from benchmark_ctrs.dataset import Dataset, DatasetWrapper

Architecture: TypeAlias = Literal["lenet", "resnet20", "resnet50", "resnet110"]


class ModelWrapper:
    """
    Wrapper class for torch modules.
    """

    norm: torch.nn.Module
    base_model: torch.nn.Module
    arch: Architecture
    dataset: Dataset

    def __init__(
        self,
        base_model: torch.nn.Module,
        norm: torch.nn.Module,
        arch: Architecture,
        dataset: Dataset,
    ):
        self.arch = arch
        self.dataset = dataset
        self.base_model = base_model
        self.model = torch.nn.Sequential(norm, base_model)

    def to(self, device: torch.device) -> Self:
        self.model = self.model.to(device)
        return self

    @staticmethod
    def get_wrapper(
        arch: Architecture,
        dataset_wrapper: DatasetWrapper,
    ) -> ModelWrapper:
        if arch == "lenet":
            base_model = lenet.LeNet(dataset_wrapper)
        elif arch == "resnet20":
            base_model = cifar_resnet.resnet(
                depth=20,
                num_classes=dataset_wrapper.classes,
            )
        elif arch == "resnet50":
            if dataset_wrapper.dataset == "imagenet":
                base_model = models.resnet50(weights=None)
            else:
                base_model = cifar_resnet.resnet(
                    depth=50,
                    num_classes=dataset_wrapper.classes,
                )
        elif arch == "resnet110":
            base_model = cifar_resnet.resnet(
                depth=110,
                num_classes=dataset_wrapper.classes,
            )

        if dataset_wrapper.dataset == "imagenet":
            base_model = torch.nn.DataParallel(base_model)
            torch.backends.cudnn.benchmark = True

        norm = Normalization(dataset_wrapper.mean, dataset_wrapper.sd)

        return ModelWrapper(base_model, norm, arch, dataset_wrapper.dataset)
