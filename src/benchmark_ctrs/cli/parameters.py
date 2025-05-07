from collections.abc import Callable, Iterable

import click
from typing_extensions import Protocol, TypeIs, TypeVar


class SupportsParameters(Protocol):
    @staticmethod
    def parameters() -> Iterable[click.Parameter]: ...


def supports_parameters(value: object) -> TypeIs[SupportsParameters]:
    return hasattr(value, "parameters")


C = TypeVar("C", bound=click.Command)


def add_parameters_from(
    source: SupportsParameters,
) -> Callable[[C], C]:
    def wrapper(command: C) -> C:
        command.params.extend(source.parameters())
        return command

    return wrapper
