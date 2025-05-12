import click

from benchmark_ctrs.cli.parameters import SupportsParameters
from benchmark_ctrs.training.cli.parameters import TrainingParametersMixin
from benchmark_ctrs.training.methods.adre import ADRETraining
from benchmark_ctrs.training.methods.custom import CustomTraining
from benchmark_ctrs.training.methods.macer import MACERTraining
from benchmark_ctrs.training.methods.noop import NoopTraining
from benchmark_ctrs.training.methods.standard import StandardTraining

__all__ = ["adre", "custom", "macer", "noop", "standard"]


def add_parameters_support(cls: type, parameters_cls: type = TrainingParametersMixin):
    return type(f"{cls.__name__}CLI", (cls, parameters_cls), {})


def noop():
    return add_parameters_support(NoopTraining)


def standard():
    return add_parameters_support(StandardTraining)


def macer():
    return add_parameters_support(MACERTraining, _MACERParameters)


class _MACERParameters(SupportsParameters):
    @staticmethod
    def parameters():
        return [
            *TrainingParametersMixin.parameters(),
            click.Option(
                ["--m-train"],
                type=click.IntRange(min=1),
                default=16,
                help="number of noise samples to use to estimate "
                "the certified radius during training",
            ),
            click.Option(
                ["--lbd"],
                type=float,
                default=16.0,
                help="regularization multiplier factor (lambda)",
            ),
            click.Option(["--beta"], type=float, default=16.0),
            click.Option(
                ["--gamma"],
                type=float,
                default=8.0,
                help="the hinge loss margin",
            ),
            click.Option(
                ["--deferred"],
                is_flag=True,
                help="apply macer only after the first learning rate drop",
            ),
        ]


def adre():
    return add_parameters_support(ADRETraining, _ADREParameters)


class _ADREParameters(SupportsParameters):
    @staticmethod
    def parameters():
        return [
            *TrainingParametersMixin.parameters(),
            click.Option(
                ["--adversarial"],
                is_flag=True,
                default=False,
                help="enables adversarial training mode",
            ),
            click.Option(
                ["-k"],
                type=click.IntRange(min=1),
                default=8,
                help="number of noise samples to use to estimate the smoothed "
                "classifier's class probabilities during training",
            ),
            click.Option(
                ["--lbd"],
                type=float,
                default=0.1,
                help="regularization multiplier factor (lambda)",
            ),
        ]


def custom():
    return add_parameters_support(CustomTraining, CustomTrainingParameters)


class CustomTrainingParameters(SupportsParameters):
    @staticmethod
    def parameters():
        return [
            *TrainingParametersMixin.parameters(),
            click.Option(
                ["--adversarial"],
                is_flag=True,
                default=False,
                help="enables adversarial training mode",
            ),
            click.Option(
                ["--max-m"],
                type=click.IntRange(min=1),
                default=8,
                help=(
                    "the maximum number of inputs as well as number of noise samples"
                    "per input to use to estimate the robust radius"
                ),
            ),
            click.Option(
                ["--lbd"],
                type=float,
                default=0.1,
                help="regularization multiplier factor (lambda)",
            ),
            click.Option(
                ["--beta"],
                type=float,
                default=16.0,
                help="The softmax temperature for the robust loss",
            ),
            click.Option(
                ["--gamma"],
                type=float,
                default=8.0,
                help="the hinge loss margin",
            ),
            click.Option(
                ["--deferred"],
                is_flag=True,
                help="apply macer only after the first learning rate drop",
            ),
        ]
