"""ModelCheckpoint that skips saving when the monitored metric is NaN or Inf."""

import math
from pathlib import Path

import lightning as L
import torch


class BestFiniteCheckpoint(L.Callback):
    """Save a checkpoint only when the monitored metric is finite and improves.

    Unlike Lightning's built-in ModelCheckpoint, this callback never writes a
    file when the metric is NaN or Inf, satisfying the requirement that no
    checkpoint exists for diverged models.
    """

    def __init__(
        self,
        monitor: str,
        dirpath: str,
        filename: str = "best_finite",
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.best = math.inf if mode == "min" else -math.inf
        self._compare = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    @property
    def best_model_path(self) -> str:
        return str(Path(self.dirpath) / f"{self.filename}.ckpt")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        val = current.item() if isinstance(current, torch.Tensor) else float(current)
        if not math.isfinite(val):
            return

        if self._compare(val, self.best):
            self.best = val
            trainer.save_checkpoint(self.best_model_path)
