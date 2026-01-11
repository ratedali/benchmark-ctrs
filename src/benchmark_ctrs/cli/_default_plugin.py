# ruff: noqa: PLC0415

from benchmark_ctrs import plugins


@plugins.hookimpl
def register_callbacks():
    from benchmark_ctrs.callbacks.certified_radius_writer import (
        CertifiedRadiusWriter,
    )

    return (CertifiedRadiusWriter,)


@plugins.hookimpl
def register_data_modules():
    from benchmark_ctrs.datasets.cifar10 import CIFAR10
    from benchmark_ctrs.datasets.imagenet import ImageNet
    from benchmark_ctrs.datasets.mnist import MNIST

    return (
        CIFAR10,
        ImageNet,
        MNIST,
    )


@plugins.hookimpl
def register_models():
    from benchmark_ctrs.modules.gaussian_aug import GaussianAug
    from benchmark_ctrs.modules.standard.cifar import CIFARStandard
    from benchmark_ctrs.modules.standard.mnist import MNISTStandard

    return (
        GaussianAug,
        CIFARStandard,
        MNISTStandard,
    )


@plugins.hookimpl
def register_lr_schedulers():
    from benchmark_ctrs.utilities import ChainedLR, GradualStepLR, SequentialLR

    return (
        ChainedLR,
        GradualStepLR,
        SequentialLR,
    )


@plugins.hookimpl
def register_certification_methods():
    from benchmark_ctrs.certification.rs_certification import RSCertification
    from benchmark_ctrs.certification.sequence.union_bound import (
        UnionBoundCertification,
    )

    return (
        RSCertification,
        UnionBoundCertification,
    )
