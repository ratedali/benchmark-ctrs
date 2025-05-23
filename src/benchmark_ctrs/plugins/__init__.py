import pluggy

hookimpl = pluggy.HookimplMarker("benchmark_ctrs")

__all__ = ["hookimpl"]
