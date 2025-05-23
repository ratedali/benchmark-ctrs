from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.core import LightningModule
from lightning.pytorch.trainer import Trainer
from typing_extensions import override

from benchmark_ctrs.metrics import certified_radius
from benchmark_ctrs.modules.module import BaseRandomizedSmoothing

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning import LightningModule, Trainer
    from torch import Tensor
    from typing_extensions import TypeIs

    from benchmark_ctrs.modules.module import Batch


FIELDS = ("idx", "label", "predict", "radius", "correct")


def _validate_module(module: Any) -> TypeIs[BaseRandomizedSmoothing]:
    return isinstance(module, BaseRandomizedSmoothing)


class CertifiedRadiusWriter(BasePredictionWriter):
    __default_params: ClassVar = certified_radius.Params()

    def __init__(
        self,
        outdir: str | None = None,
        filename: str = "cert.csv",
        params: certified_radius.Params | None = None,
    ) -> None:
        super().__init__(write_interval="batch")
        self._outdir = Path(outdir) if outdir else None
        if self._outdir is not None and not self._outdir.is_dir():
            raise ValueError(f"{outdir} is not a directory.")

        self._filename = filename
        self._params = params if params else CertifiedRadiusWriter.__default_params

    @override
    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not _validate_module(pl_module):
            raise TypeError(
                "Only modules that are subclasses of "
                f"{BaseRandomizedSmoothing.__qualname__} are supported"
            )
        path = self._resolve_output_path(trainer)
        with path.open("at") as f:
            writer = DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

    @override
    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not _validate_module(pl_module):
            raise TypeError(
                "Only modules that are subclasses of "
                f"{BaseRandomizedSmoothing.__qualname__} are supported"
            )
        if batch_indices is None:
            raise ValueError("Batch indices is required")

        path = self._resolve_output_path(trainer)
        self._certify(path, pl_module, batch, list(batch_indices))

    def _certify(
        self,
        path: Path,
        pl_module: BaseRandomizedSmoothing,
        batch: Batch,
        batch_indices: list[int],
    ) -> None:
        inputs, targets = batch
        cr = certified_radius.CertifiedRadius(
            pl_module.base_classifier,
            self._params,
            num_classes=pl_module.num_classes,
            sigma=pl_module.sigma,
            reduction="none",
        )

        cr.update(inputs)
        indices, predictions, radii = cast("tuple[Tensor,...]", cr.compute())
        indices, predictions, radii = (
            indices.view(-1),
            predictions.view(-1),
            radii.view(-1),
        )

        cr.reset()
        if indices.numel() > 0:
            with path.open("at") as f:
                writer = DictWriter(f, fieldnames=FIELDS)
                writer.writerows(
                    {
                        "idx": batch_indices[i],
                        "label": int(targets[i].item()),
                        "predict": int(predictions[i].item()),
                        "radius": float(radii[i].item()),
                        "correct": 1 if predictions[i] == targets[i] else 0,
                    }
                    for i in indices
                )

    def _resolve_output_path(self, trainer: Trainer) -> Path:
        if self._outdir:
            return self._outdir / self._filename

        outdir = Path(trainer.default_root_dir)
        if len(trainer.loggers) > 0:
            logger = trainer.loggers[0]
            if logger.save_dir is not None:
                outdir = Path(logger.save_dir)
            name = logger.name if logger.name else "cert_logs"
            version = logger.version
            version = version if isinstance(version, str) else f"version_{version}"
            outdir = outdir / name / version
        return outdir / self._filename
