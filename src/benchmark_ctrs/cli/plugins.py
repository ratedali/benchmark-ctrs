from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pluggy

from benchmark_ctrs.cli import _default_plugin
from benchmark_ctrs.plugins import spec

if TYPE_CHECKING:
    from benchmark_ctrs.plugins.spec import HookType


def get_hook() -> HookType:
    pm = pluggy.PluginManager("benchmark_ctrs")
    pm.add_hookspecs(spec)
    pm.load_setuptools_entrypoints("benchmark_ctrs")
    pm.register(_default_plugin)
    return cast("HookType", pm.hook)
