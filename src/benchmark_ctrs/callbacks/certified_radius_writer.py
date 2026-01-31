from collections.abc import Sequence
from csv import DictWriter
from pathlib import Path
from typing import Any, Final, cast

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override

from benchmark_ctrs.modules import Batch, PredictionResult

__all__ = ["CERT_FIELDS", "CLEAN_FIELDS", "CertifiedRadiusWriter"]

CERT_FIELDS: Final = ("idx", "label", "predict", "radius", "correct")
CLEAN_FIELDS: Final = ("idx", "label", "predict", "correct")


class CertifiedRadiusWriter(BasePredictionWriter):
    def __init__(
        self,
        outdir: str | None = None,
        filename: str | None = "cert.csv",
        clean_filename: str | None = "clean.csv",
        *,
        overwrite: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")
        self._outdir = Path(outdir) if outdir else None
        if self._outdir is not None and not self._outdir.is_dir():
            raise ValueError(f"{outdir} is not a directory.")

        self.filenames = {
            "cert": (filename, CERT_FIELDS),
            "clean": (clean_filename, CLEAN_FIELDS),
        }
        self.overwrite = overwrite

    @override
    @rank_zero_only
    def on_predict_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        for filename, header in self.filenames.values():
            if filename:
                path = self._resolve_output_path(trainer, filename)

                if path.exists() and not self.overwrite:
                    raise ValueError(f"{path} exists, and overwite=False")

                with path.open("tw") as f:
                    writer = DictWriter(f, fieldnames=header)
                    writer.writeheader()

    @override
    @rank_zero_only
    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction: Any | None,
        batch_indices: Sequence[int] | None,
        batch: Batch,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(prediction, dict) or "clean" not in prediction:
            raise TypeError(
                "`predict_step` should return `benchmark_ctrs.modules.PredictionResult`"
                " instance from global rank 0."
            )
        prediction = cast("PredictionResult", prediction)

        if batch_indices is None:
            raise ValueError("Batch indices is required")

        batch_indices = list(batch_indices)
        _inputs, targets = batch
        targets = cast("list[int]", targets.long().tolist())
        predictions: list[float] = prediction["clean"].view(-1).tolist()

        clean_filename, clean_header = self.filenames["clean"]
        if clean_filename:
            clean_path = self._resolve_output_path(trainer, clean_filename)

            if len(predictions) > 0:
                with clean_path.open("at") as f:
                    writer = DictWriter(f, fieldnames=clean_header)
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

        cert_filename, cert_header = self.filenames["cert"]
        if cert_filename and "certification" in prediction:
            cert_path = self._resolve_output_path(trainer, cert_filename)

            cert = prediction["certification"]
            indices: list[int] = cert.indices.view(-1).long().tolist()
            predictions: list[float] = cert.predictions.view(-1).tolist()
            radii: list[float] = cert.radii.view(-1).tolist()

            if len(indices) > 0:
                with cert_path.open("at") as f:
                    writer = DictWriter(f, fieldnames=cert_header)
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
