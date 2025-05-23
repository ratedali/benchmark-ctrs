from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark_ctrs import plugins

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lightning as L

    from benchmark_ctrs.datasets.module import BaseDataModule
    from benchmark_ctrs.modules.module import BaseRandomizedSmoothing


@plugins.hookimpl
def register_callbacks() -> Sequence[type[L.Callback]]:
    from benchmark_ctrs.callbacks.certified_radius_writer import (
        CertifiedRadiusWriter,
    )

    return (CertifiedRadiusWriter,)


@plugins.hookimpl
def register_data_modules() -> Sequence[type[BaseDataModule]]:
    from benchmark_ctrs.datasets.cifar10 import CIFAR10
    from benchmark_ctrs.datasets.imagenet import ImageNet
    from benchmark_ctrs.datasets.mnist import MNIST

    return (
        CIFAR10,
        ImageNet,
        MNIST,
    )


@plugins.hookimpl
def register_models() -> Sequence[type[BaseRandomizedSmoothing]]:
    from benchmark_ctrs.modules.gaussian_aug import GaussianAug

    return (GaussianAug,)
