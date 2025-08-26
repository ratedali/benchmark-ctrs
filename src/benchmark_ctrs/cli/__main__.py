from __future__ import annotations

import logging
from pathlib import Path

from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from typing_extensions import override

import benchmark_ctrs
from benchmark_ctrs.cli import plugins
from benchmark_ctrs.datasets.imagenet import ImageNet
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
        hook.register_models()
    except Exception:
        logger.exception("Error raised when registering plugin training modules.")

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
            "dump_header": [f"# benchmark-ctrs=={benchmark_ctrs.__version__}"],
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
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
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
