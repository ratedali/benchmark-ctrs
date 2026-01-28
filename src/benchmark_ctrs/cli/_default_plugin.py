# ruff: noqa: PLC0415

from benchmark_ctrs import plugins


@plugins.hookimpl
def register_callbacks():
    from benchmark_ctrs.callbacks import (
        CertifiedRadiusWriter,
    )

    return (CertifiedRadiusWriter,)


@plugins.hookimpl
def register_data_modules():
    from benchmark_ctrs.datasets import CIFAR10, MNIST, ImageNet

    return (
        CIFAR10,
        ImageNet,
        MNIST,
    )


@plugins.hookimpl
def register_models():
    from benchmark_ctrs.modules import (
        CIFARStandard,
        GaussianAug,
        ImageNetStandard,
        MNISTStandard,
    )

    return (
        GaussianAug,
        CIFARStandard,
        ImageNetStandard,
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
    from benchmark_ctrs.certification import (
        BettingCertification,
        RSCertification,
        UBCertification,
    )

    return (
        RSCertification,
        BettingCertification,
        UBCertification,
    )
