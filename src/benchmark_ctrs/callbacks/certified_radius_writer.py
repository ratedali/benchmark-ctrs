from collections.abc import Collection, Iterable, Mapping, Sequence
from csv import DictWriter
from pathlib import Path
from typing import Any, Final, cast

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
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

        self.overwrite = overwrite

        self.cert_output = (filename, CERT_FIELDS) if filename else None
        self.clean_output = (clean_filename, CLEAN_FIELDS) if clean_filename else None

    @property
    def outputs(self) -> list[tuple[str, Collection[str]]]:
        outputs = []
        if self.cert_output:
            outputs.append(self.cert_output)
        if self.clean_output:
            outputs.append(self.clean_output)
        return outputs

    @override
    def on_predict_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        for filename, header in self.outputs:
            path = self._resolve_output_path(trainer, filename)

            if path.exists() and not self.overwrite:
                raise RuntimeError(f"{path} already exists, and overwite=False")

            if trainer.global_rank == 0:
                with path.open("tw") as f:
                    writer = DictWriter(f, fieldnames=header)
                    writer.writeheader()

    @override
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
        if prediction is None:
            return

        if not isinstance(prediction, dict) or "clean" not in prediction:
            raise TypeError(
                "`predict_step` should return a dict which is a "
                "`benchmark_ctrs.modules.PredictionResult` instance"
            )
        prediction = cast("PredictionResult", prediction)

        if batch_indices is None:
            raise ValueError("Batch indices are required")

        batch_indices = list(batch_indices)
        _inputs, targets = batch[:2]

        targets = cast("list[int]", targets.long().tolist())
        predictions: list[float] = prediction["clean"].view(-1).tolist()

        clean_rows = [
            {
                "idx": batch_indices[i],
                "label": targets[i],
                "predict": int(pred),
                "correct": 1 if pred == targets[i] else 0,
            }
            for i, pred in enumerate(predictions)
        ]
        self._write_rows(trainer, clean_rows, self.clean_output)

        if "certification" in prediction:
            cert = prediction["certification"]
            indices: list[int] = cert.indices.view(-1).long().tolist()
            predictions: list[float] = cert.predictions.view(-1).tolist()
            radii: list[float] = cert.radii.view(-1).tolist()

            cert_rows = [
                {
                    "idx": batch_indices[batch_idx],
                    "label": targets[batch_idx],
                    "predict": predictions[i],
                    "radius": radii[i],
                    "correct": int(predictions[i] == targets[batch_idx]),
                }
                for i, batch_idx in enumerate(indices)
            ]
            self._write_rows(trainer, cert_rows, self.cert_output)

    def _write_rows(
        self,
        trainer: L.Trainer,
        rows: Iterable[Mapping[str, Any]],
        output: tuple[str, Collection[str]] | None,
    ) -> None:
        if output is None:
            return
        filename, header = output
        cert_path = self._resolve_output_path(trainer, filename)

        with cert_path.open("at") as f:
            writer = DictWriter(f, fieldnames=header)
            writer.writerows(rows)

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
