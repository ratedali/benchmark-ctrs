from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from typing_extensions import override

from benchmark_ctrs import datasets, modules
from benchmark_ctrs.datasets import *  # noqa: F403
from benchmark_ctrs.modules import *  # noqa: F403


def main():
    BenchmarkCTRSCLI(
        model_class=modules.RandomizedSmoothing,
        subclass_mode_model=True,
        datamodule_class=datasets.ClassificationDataModule,
        subclass_mode_data=True,
        parser_kwargs={
            "default_env": True,
            "fit": {
                "default_config_files": [
                    Path(__file__).parent / "default_config_fit.yml"
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
            "data.dataset",
            "model.init_args.is_imagenet",
            compute_fn=lambda dataset: dataset == datasets.ImageNet,
            apply_on="instantiate",
        )


if __name__ == "__main__":
    main()
