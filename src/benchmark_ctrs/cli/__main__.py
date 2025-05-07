import logging
import sys

import click

from benchmark_ctrs.training.cli import train

_logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", is_flag=True)
def main(verbose: bool):  # noqa: FBT001
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not rootLogger.handlers:
        simple_fmt = logging.Formatter("%(message)s")
        verbose_fmt = logging.Formatter(
            "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        )

        info_handler = logging.StreamHandler(sys.stdout)
        info_handler.setFormatter(simple_fmt if not verbose else verbose_fmt)
        info_handler.setLevel(logging.INFO)
        rootLogger.addHandler(info_handler)

        if verbose:
            verbose_handler = logging.StreamHandler(sys.stderr)
            verbose_handler.setFormatter(verbose_fmt)
            verbose_handler.setLevel(logging.DEBUG)
            rootLogger.addHandler(verbose_handler)

    _logger.debug("command executed")


main.add_command(train)


if __name__ == "__main__":
    main()
