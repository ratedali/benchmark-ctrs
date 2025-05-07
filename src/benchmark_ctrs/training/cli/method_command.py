import logging
from typing import Generic, cast

import click
from typing_extensions import TypeVar

from benchmark_ctrs.training.methods import TrainingMethod
from benchmark_ctrs.training.parameters import TrainingParameters
from benchmark_ctrs.training.run import TrainingRun

_logger = logging.getLogger(__name__)


_Tparams = TypeVar("_Tparams", bound=TrainingParameters)


class TrainingMethodCommand(click.Command, Generic[_Tparams]):
    def __init__(self, *args, method_cls: type[TrainingMethod[_Tparams]], **kwargs):
        context_settings: dict = kwargs.get("context_settings", {})
        context_settings["show_default"] = True
        super().__init__(
            *args,
            **kwargs,
            context_settings=context_settings,
            callback=self.make_callback(method_cls),
        )

    @classmethod
    def make_callback(cls, method_cls: type[TrainingMethod[_Tparams]]):
        def callback(**params):
            _logger.debug(
                "Executing training command for method class %s",
                cls.__name__,
            )
            _logger.debug("Parsed parameters: %r", params)
            params = cast("_Tparams", params)
            method = method_cls.create_instance(params)
            training = TrainingRun(params["id"], method, params)
            training.run()

        return callback
