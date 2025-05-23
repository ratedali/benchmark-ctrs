from __future__ import annotations

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lightning as L

    from benchmark_ctrs.datasets.module import BaseDataModule
    from benchmark_ctrs.modules.module import BaseRandomizedSmoothing

    class HookType:
        @staticmethod
        def register_callbacks() -> Sequence[type[L.Callback]]:
            """
            Return all lightning callback types that should be accessible to the CLI
            """
            ...

        @staticmethod
        def register_data_modules() -> Sequence[type[BaseDataModule]]:
            """
            Return all data module types that should be accessible to the CLI
            """
            ...

        @staticmethod
        def register_models() -> Sequence[type[BaseRandomizedSmoothing]]:
            """
            Return all lightning module types that should be accessible to the CLI
            """
            ...


hookspec = pluggy.HookspecMarker("benchmark_ctrs")


@hookspec
def register_callbacks() -> Sequence[type[L.Callback]]:
    """Return all lightning callback types that should be accessible to the CLI"""
    ...  # noqa: PIE790


@hookspec
def register_data_modules() -> Sequence[type[BaseDataModule]]:
    """Return all data module types that should be accessible to the CLI"""
    ...  # noqa: PIE790


@hookspec
def register_models() -> Sequence[type[BaseRandomizedSmoothing]]:
    """Return all lightning module types that should be accessible to the CLI"""
    ...  # noqa: PIE790
