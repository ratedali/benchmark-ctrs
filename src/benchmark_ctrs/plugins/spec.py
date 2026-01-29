# ruff: noqa: PIE790
from collections.abc import Iterable

import lightning as L
import pluggy

from benchmark_ctrs.certification.abc import CertificationMethod
from benchmark_ctrs.criterions import Criterion
from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.modules.module import BaseModule
from benchmark_ctrs.types import LRScheduler, LRSchedulerCallable


class HookType:
    @staticmethod
    def register_callbacks() -> Iterable[type[L.Callback]]:
        """
        Return all lightning callback types that should be accessible to the CLI
        """
        ...

    @staticmethod
    def register_data_modules() -> Iterable[type[BaseDataModule]]:
        """
        Return all data module types that should be accessible to the CLI
        """
        ...

    @staticmethod
    def register_models() -> Iterable[type[BaseModule]]:
        """
        Return all lightning module types that should be accessible to the CLI
        """
        ...

    @staticmethod
    def register_certification_methods() -> Iterable[type[CertificationMethod]]:
        """
        Return all certification methods that should be accessible to the CLI
        """
        ...

    @staticmethod
    def register_criterions() -> Iterable[type[Criterion]]:
        """
        Return all criterions that should be accessible to the CLI
        """
        ...

    @staticmethod
    def register_lr_schedulers() -> Iterable[type[LRScheduler] | LRSchedulerCallable]:
        """
        Return all learning rate scheduler classes and factories types that
        should be accessible to the CLI
        """
        ...


hookspec = pluggy.HookspecMarker("benchmark_ctrs")


@hookspec
def register_callbacks() -> Iterable[type[L.Callback]]:
    """Return all lightning callback types that should be accessible to the CLI"""
    ...


@hookspec
def register_data_modules() -> Iterable[type[BaseDataModule]]:
    """Return all data module types that should be accessible to the CLI"""
    ...


@hookspec
def register_models() -> Iterable[type[BaseModule]]:
    """Return all lightning module types that should be accessible to the CLI"""
    ...


@hookspec
def register_certification_methods() -> Iterable[type[CertificationMethod]]:
    """
    Return all certification methods that should be accessible to the CLI
    """
    ...


@hookspec
def register_criterions() -> Iterable[type[Criterion]]:
    """
    Return all criterions that should be accessible to the CLI
    """
    ...


@hookspec
def register_lr_schedulers() -> Iterable[type[LRScheduler] | LRSchedulerCallable]:
    """
    Return all learning rate scheduler classes and factories types that
    should be accessible to the CLI
    """
    ...
