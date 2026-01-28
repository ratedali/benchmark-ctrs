from collections.abc import Sequence
from csv import DictWriter
from pathlib import Path
from typing import Any, Final, cast

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from typing_extensions import override

from benchmark_ctrs.modules.module import BaseModule, Batch, PredictionResult

__all__ = ["CERT_FIELDS", "CLEAN_FIELDS", "CertifiedRadiusWriter"]

CERT_FIELDS: Final = ("idx", "label", "predict", "radius", "correct")
CLEAN_FIELDS: Final = ("idx", "label", "predict", "correct")


class CertifiedRadiusWriter(BasePredictionWriter):
    def __init__(
        self,
        outdir: str | None = None,
        filename: str = "cert.csv",
        clean_filename: str = "clean.csv",
        *,
        ignore_cert: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")
        self._outdir = Path(outdir) if outdir else None
        if self._outdir is not None and not self._outdir.is_dir():
            raise ValueError(f"{outdir} is not a directory.")

        self._filename = filename
        self._clean_filename = clean_filename
        self._ignore_cert = ignore_cert

    @override
    def on_predict_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not isinstance(pl_module, BaseModule):
            raise TypeError(
                "Only modules that are subclasses of "
                f"{BaseModule.__qualname__} are supported"
            )

        clean_path = self._resolve_output_path(trainer, self._clean_filename)
        with clean_path.open("tw") as f:
            writer = DictWriter(f, fieldnames=CLEAN_FIELDS)
            writer.writeheader()

        if not self._ignore_cert:
            cert_path = self._resolve_output_path(trainer, self._filename)
            with cert_path.open("tw") as f:
                writer = DictWriter(f, fieldnames=CERT_FIELDS)
                writer.writeheader()

    @override
    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction: PredictionResult,
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
        if not isinstance(prediction, dict):
            raise TypeError("return type of `predict_step` should not be None")
        if batch_indices is None:
            raise ValueError("Batch indices is required")

        batch_indices = list(batch_indices)
        _inputs, targets = batch
        targets = cast("list[int]", targets.long().tolist())
        predictions: list[float] = prediction["clean"].view(-1).tolist()

        clean_path = self._resolve_output_path(trainer, self._clean_filename)

        if len(predictions) > 0:
            with clean_path.open("at") as f:
                writer = DictWriter(f, fieldnames=CLEAN_FIELDS)
                writer.writerows(
                    [
                        {
                            "idx": batch_indices[i],
                            "label": targets[i],
                            "predict": int(pred),
                            "correct": 1 if pred == targets[i] else 0,
                        }
                        for i, pred in enumerate(predictions)
                    ]
                )

        if "certification" in prediction and not self._ignore_cert:
            cert_path = self._resolve_output_path(trainer, self._filename)
            cert = prediction["certification"]
            indices: list[int] = cert.indices.view(-1).long().tolist()
            predictions: list[float] = cert.predictions.view(-1).tolist()
            radii: list[float] = cert.radii.view(-1).tolist()

            if len(indices) > 0:
                with cert_path.open("at") as f:
                    writer = DictWriter(f, fieldnames=CERT_FIELDS)
                    writer.writerows(
                        [
                            {
                                "idx": batch_indices[i],
                                "label": targets[i],
                                "predict": predictions[i],
                                "radius": radii[i],
                                "correct": int(predictions[i] == targets[i]),
                            }
                            for i in indices
                        ]
                    )

    def _resolve_output_path(self, trainer: L.Trainer, filename: str) -> Path:
        if self._outdir:
            return self._outdir / filename

        outdir = Path(trainer.default_root_dir)
        if len(trainer.loggers) > 0:
            logger = trainer.loggers[0]
            if logger.save_dir is not None:
                outdir = Path(logger.save_dir)
            name = logger.name if logger.name else "cert_logs"
            version = logger.version
            version = version if isinstance(version, str) else f"version_{version or 0}"
            outdir = outdir / name / version
        return outdir / filename
