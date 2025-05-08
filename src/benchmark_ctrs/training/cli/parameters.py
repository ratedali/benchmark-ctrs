import pathlib

import click

from benchmark_ctrs.cli.parameters import SupportsParameters


class TrainingParametersMixin(SupportsParameters):
    @staticmethod
    def parameters() -> list[click.Parameter]:
        return [
            click.Argument(
                ["dataset"],
                type=click.Choice(("mnist", "cifar10", "imagenet")),
            ),
            click.Argument(
                ["architecture"],
                type=click.Choice(("lenet", "resnet50", "resnet110")),
            ),
            click.Option(["--id"], type=int, required=True, help="run ID"),
            click.Option(
                ["--noise-sd"],
                type=click.FloatRange(min=0, max=1),
                required=True,
                help="std of the noise normal distribution used by the smoothed "
                "classifier (standard values are: 0.00, 0.25, 0.50, 1.00)",
            ),
            click.Option(
                ["--epochs"],
                type=click.IntRange(min=1),
                default=150,
                help="number of total epochs to run",
            ),
            click.Option(
                ["--batch-size"],
                type=click.IntRange(min=1),
                default=400,
                help="training batch size",
            ),
            click.Option(
                ["--num-workers"],
                type=click.IntRange(min=0),
                default=2,
                help="number of data loading workers",
            ),
            click.Option(
                ["--optimizer"],
                type=click.Choice(("sgd",)),
                default="sgd",
                help="optimizer used for training",
            ),
            click.Option(
                ["--loss"],
                type=click.Choice(("cross-entropy",)),
                default="cross-entropy",
                help="loss function to optimize",
            ),
            click.Option(
                ["--validation"],
                type=click.Choice(("kfold", "set")),
                default="set",
                help="validation method if any",
            ),
            click.Option(
                ["--validation-set-split"],
                type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
                default=0.2,
                help="proportion of the validations set split to the entire training "
                'set, when using "set" validation',
            ),
            click.Option(
                ["--validation-kfold-splits"],
                type=click.IntRange(min=2),
                default=5,
                help='number of splits when using "kfold" validation',
            ),
            click.Option(
                ["--lr"],
                type=click.FloatRange(min=0.0, min_open=True),
                default=0.01,
                help="initial learning rate",
            ),
            click.Option(
                ["--lr-schedule"],
                type=click.Choice(("step", "constant")),
                default="step",
                help="learning rate schedule type",
            ),
            click.Option(
                ["--lr-schedule-gamma"],
                type=click.FloatRange(min=0.0, min_open=True),
                default=0.1,
                help="update factor for the learning rate schedule",
            ),
            click.Option(
                ["--lr-step-size"],
                type=click.IntRange(min=1),
                default=50,
                help="step size when using the step learning rate schedule",
            ),
            click.Option(
                ["--weight-decay"],
                type=float,
                default=1e-4,
                help="weight decay factor",
            ),
            click.Option(
                ["--momentum"],
                type=float,
                default=0.9,
                help="momentum weight",
            ),
            click.Option(
                ["--save/--no-save"],
                is_flag=True,
                default=True,
                help="whether training checkpoints, logs, and metrics "
                "are saved to rundir",
            ),
            click.Option(
                ["--rundir"],
                type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
                default="runs",
                help="path to save the latest model checkpoint during training",
            ),
            click.Option(
                ["--resume / --no-resume"],
                is_flag=True,
                default=False,
                help="whether training should resume from "
                "a previously saved checkpoint",
            ),
            click.Option(
                ["--resume-path"],
                type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
                help="path for a model training checkpoint to resume from, "
                "if not provided but resume is used the rundir will be used",
            ),
            click.Option(
                ["--log-freq"],
                type=int,
                default=5,
                help="log training metrics this many per epoch",
            ),
            click.Option(
                ["--data-dir"],
                type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
                envvar="DATASET_CACHE",
                default=pathlib.Path(".", "datasets"),
                help="directory to save downloaded datasets",
            ),
            click.Option(
                ["--log-grads"],
                is_flag=True,
                default=False,
                help="log the L2 gradient norms during training as a metric",
            ),
            click.Option(["--profiling / --no-profiling"], is_flag=True, default=False),
        ]
