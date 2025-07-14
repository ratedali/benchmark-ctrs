from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.core import LightningModule
from lightning.pytorch.trainer import Trainer
from typing_extensions import override

from benchmark_ctrs.modules.module import BaseModule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning import LightningModule, Trainer

    from benchmark_ctrs.metrics.certified_radius import CertificationResult
    from benchmark_ctrs.modules.module import Batch


FIELDS: Final = ("idx", "label", "predict", "radius", "correct")


class CertifiedRadiusWriter(BasePredictionWriter):
    def __init__(self, outdir: str | None = None, filename: str = "cert.csv") -> None:
        super().__init__(write_interval="batch")
        self._outdir = Path(outdir) if outdir else None
        if self._outdir is not None and not self._outdir.is_dir():
            raise ValueError(f"{outdir} is not a directory.")

        self._filename = filename

    @override
    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not isinstance(pl_module, BaseModule):
            raise TypeError(
                "Only modules that are subclasses of "
                f"{BaseModule.__qualname__} are supported"
            )
        path = self._resolve_output_path(trainer)
        with path.open("tw") as f:
            writer = DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

    @override
    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: CertificationResult | None,
        batch_indices: Sequence[int] | None,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(pl_module, BaseModule):
            raise TypeError(
                "Only modules that are subclasses of "
                f"{BaseModule.__qualname__} are supported"
            )
        if not prediction:
            raise TypeError(
                "return type of `predict_step` should be `CertificationResult`"
            )

        if batch_indices is None:
            raise ValueError("Batch indices is required")
        batch_indices = list(batch_indices)

        path = self._resolve_output_path(trainer)

        _inputs, targets = batch

        indices = prediction.indices.view(-1)
        predictions = prediction.predictions.view(-1)
        radii = prediction.radii.view(-1)

        if prediction.indices.numel() > 0:
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
            version = version if isinstance(version, str) else f"version_{version or 0}"
            outdir = outdir / name / version
        return outdir / self._filename
