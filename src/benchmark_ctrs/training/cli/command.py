import logging
import sys
from typing import Callable, cast

import click
from typing_extensions import TypeAlias, TypeVar

import benchmark_ctrs.training.cli.methods as builtin_methods
from benchmark_ctrs.cli.parameters import add_parameters_from, supports_parameters
from benchmark_ctrs.training.cli.method_command import TrainingMethodCommand
from benchmark_ctrs.training.methods.abc import TrainingMethod
from benchmark_ctrs.training.parameters import TrainingParameters

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


_logger = logging.getLogger(__name__)

_Tparams = TypeVar("_Tparams", bound=TrainingParameters)
LoadImpl: TypeAlias = Callable[[], type[TrainingMethod[_Tparams]]]


class TrainCLI(click.MultiCommand):
    def __init__(
        self,
        *args,
        plugins_entrypoint="benchmark_ctrs.cli.training",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        _logger.debug("Detecting and loading plugins")
        self._plugins = dict(self._find_plugins(plugins_entrypoint))

    def list_commands(self, ctx: click.Context):
        base = super().list_commands(ctx)
        builtin = set(builtin_methods.__all__)
        plugins = self._plugins.keys()
        return base + sorted(builtin.union(plugins))

    def get_command(self, ctx: click.Context, cmd_name: str):
        loader = None
        if hasattr(builtin_methods, cmd_name):
            loader = cast(
                "LoadImpl[TrainingParameters]",
                getattr(builtin_methods, cmd_name),
            )
        elif cmd_name in self._plugins:
            loader = self._plugins[cmd_name]
        else:
            return super().get_command(ctx, cmd_name)

        method_cls = TrainCLI._load_method_cls(loader)
        return TrainCLI._get_plugin_command(cmd_name, method_cls)

    @classmethod
    def _get_plugin_command(cls, name: str, method_cls):
        cmd = TrainingMethodCommand(
            name=name,
            method_cls=method_cls,
        )
        return add_parameters_from(method_cls)(cmd)

    @classmethod
    def _find_plugins(cls, entrypoint):
        for plugin in entry_points(group=entrypoint):
            load_impl: LoadImpl[TrainingParameters] = plugin.load()
            if not callable(load_impl):
                _logger.warning(
                    "Skipping plugin %s (%s): entrypoint must be callable.",
                    plugin.name,
                    plugin.value,
                )
                continue

            yield plugin.name, load_impl

    @classmethod
    def _load_method_cls(cls, load_impl: LoadImpl[_Tparams]):
        method_cls = load_impl()
        if not isinstance(method_cls, type) or not issubclass(
            method_cls,
            TrainingMethod,
        ):
            raise TypeError("Plugin must be return subtype of TrainingMethod.")

        if not supports_parameters(method_cls):
            raise ValueError(
                "Plugin must return a type implementing "
                "the SupportsParameters protocol.",
            )
        return method_cls


@click.command(cls=TrainCLI)
def train():
    pass
