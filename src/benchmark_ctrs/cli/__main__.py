import logging
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torchvision
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from typing_extensions import override

import benchmark_ctrs
from benchmark_ctrs.cli import plugins
from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.modules.module import BaseModule

logger = logging.getLogger(__name__)


def main(args: ArgsType = None) -> None:
    hook = plugins.get_hook()

    try:
        hook.register_callbacks()
    except Exception:
        logger.exception("Error raised when registering plugin callbacks.")

    try:
        hook.register_data_modules()
    except Exception:
        logger.exception("Error raised when registering plugin data modules.")

    try:
        hook.register_certification_methods()
    except Exception:
        logger.exception("Error raised when registering plugin certification methods.")

    try:
        hook.register_models()
    except Exception:
        logger.exception("Error raised when registering plugin training modules.")

    try:
        hook.register_criterions()
    except Exception:
        logger.exception("Error raised when registering plugin criterions.")

    try:
        hook.register_lr_schedulers()
    except Exception:
        logger.exception(
            "Error raised when registering plugin learning rate schedulers."
        )

    BenchmarkCTRSCLI(
        model_class=BaseModule,
        subclass_mode_model=True,
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
        save_config_kwargs={
            "overwrite": True,
            "multifile": True,
        },
        parser_kwargs={
            "version": benchmark_ctrs.__version__,
            "default_env": True,
            "fit": {
                "default_config_files": [
                    Path(__file__).parent / "default_config_fit.yml"
                ]
            },
            "predict": {
                "default_config_files": [
                    Path(__file__).parent / "default_config_predict.yml"
                ]
            },
        },
        args=args,
    )


class BenchmarkCTRSCLI(LightningCLI):
    @override
    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        kwargs.setdefault(
            "dump_header",
            [
                f"torch=={torch.__version__}",
                f"torchvision=={torchvision.__version__}",
                f"lightning.pytorch=={pl.__version__}",  # type: ignore
                f"benchmark-ctrs=={benchmark_ctrs.__version__}",
            ],
        )
        return super().init_parser(**kwargs)

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        # Easy way to overwrite name and version when using multiple loggers
        parser.add_argument(
            "--name",
            type=str | None,
            default="lightning_logs",
            help="A name to identify identify similar runs",
        )
        parser.add_argument(
            "--version",
            type=str | None,
            default=None,
            help="A version number/name to identify this run",
        )

        parser.link_arguments(
            "trainer.default_root_dir", "trainer.logger.init_args.save_dir"
        )
        parser.link_arguments("name", "trainer.logger.init_args.name")
        parser.link_arguments("version", "trainer.logger.init_args.version")

        # Link values from the data module to the training module
        parser.link_arguments(
            "data.classes", "model.init_args.num_classes", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.mean", "model.init_args.mean", apply_on="instantiate"
        )
        parser.link_arguments("data.std", "model.init_args.std", apply_on="instantiate")
        parser.link_arguments(
            "data.default_arch",
            "model.init_args.default_arch",
            apply_on="instantiate",
        )


if __name__ == "__main__":
    main()
