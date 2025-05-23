from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from typing_extensions import override

import benchmark_ctrs
from benchmark_ctrs.cli import plugins
from benchmark_ctrs.datasets.imagenet import ImageNet
from benchmark_ctrs.datasets.module import BaseDataModule
from benchmark_ctrs.modules.module import BaseRandomizedSmoothing


def main() -> None:
    hook = plugins.get_hook()
    hook.register_data_modules()
    hook.register_models()
    hook.register_callbacks()

    BenchmarkCTRSCLI(
        model_class=BaseRandomizedSmoothing,
        subclass_mode_model=True,
        datamodule_class=BaseDataModule,
        subclass_mode_data=True,
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
    )


class BenchmarkCTRSCLI(LightningCLI):
    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # Link values from the data module to the training module
        parser.link_arguments(
            "data.classes", "model.init_args.num_classes", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.means", "model.init_args.means", apply_on="instantiate"
        )
        parser.link_arguments("data.sds", "model.init_args.sds", apply_on="instantiate")
        parser.link_arguments(
            "data.default_arch",
            "model.init_args.arch",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data",
            "model.init_args.is_imagenet",
            compute_fn=lambda dataset: isinstance(dataset, ImageNet),
            apply_on="instantiate",
        )


if __name__ == "__main__":
    main()
