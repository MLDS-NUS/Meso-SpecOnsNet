from typing import Any

import einops
import rootutils
import torch
from lightning import LightningModule
from torch import nn, optim
from torchmetrics import MeanMetric, MetricCollection, MinMetric

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import RankedLogger  # noqa: E402

logger = RankedLogger(__name__, rank_zero_only=True)


class MesoLitModule(LightningModule):
    """Lightning Module for mesoscopic dynamics learning."""

    def __init__(
        self,
        dynamics: nn.Module,
        optimizer: optim.Optimizer,
        dt: float,
        compile: bool,
        drift: nn.Module | None = None,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        accumulated_nsteps: int = 1,
        detach_probability: float = 0.0,  # run 366 vs 346
        intermediate_losses: bool = True,
        loss_fn: str = "mse",
        reg_weight: float | None = None,
    ) -> None:
        super().__init__()

        # Store init parameters for checkpointing
        self.save_hyperparameters(logger=False, ignore=["dynamics", "drift"])

        self.dynamics = dynamics
        self.drift = drift
        self.dt = dt

        # Loss function
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_fn}")

        # Metrics for tracking training progress
        self.train_loss = MetricCollection(
            {
                "loss": MeanMetric(),
                "pred_loss": MeanMetric(),
            }
            | {f"pred_loss_{k + 1}": MeanMetric() for k in range(self.hparams.accumulated_nsteps)}
            | {f"reg_loss_{k + 1}": MeanMetric() for k in range(self.hparams.accumulated_nsteps)}
        )
        self.valid_loss = MetricCollection(
            {
                "loss": MeanMetric(),
                "pred_loss": MeanMetric(),
            }
            | {f"pred_loss_{k + 1}": MeanMetric() for k in range(self.hparams.accumulated_nsteps)}
            | {f"reg_loss_{k + 1}": MeanMetric() for k in range(self.hparams.accumulated_nsteps)}
        )

        # Best validation metrics
        self.valid_loss_best = MinMetric()

        self._is_compiled = False

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x_sequence: Input sequence tensor of shape (b, nsteps, c, *domain_shape)
        :return: Predicted next state of shape (b, nsteps, c, *domain_shape)
        """
        b, seq_len, *_ = x_seq.shape
        x_src = einops.rearrange(x_seq, "b L ... -> (b L) ...")
        out = self.dynamics.forward_step(x_src, dt=self.dt)  # (b, (-1,), ...)
        return out.reshape(b, seq_len, *out.shape[1:])  # (b, L1, ...)

    def regularization_loss(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss if defined in dynamics.

        :param x_seq: Input sequence tensor of shape (b, nsteps, c, *domain_shape)
        :return: Regularization loss scalar
        """
        if hasattr(self.dynamics, "regularization_loss"):
            b, seq_len, *_ = x_seq.shape
            x_src = einops.rearrange(x_seq, "b L ... -> (b L) ...")
            reg = self.dynamics.regularization_loss(x_src)
            return reg.mean()
        return torch.tensor(0.0, device=x_seq.device)

    def model_step(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        acc_nsteps = int(self.hparams.accumulated_nsteps)

        x_seq = seq[:, :-acc_nsteps]

        step2loss = {}
        loss = 0.0

        for k in range(acc_nsteps):
            # 1. Forward step (Autoregressive)
            x_seq = self.forward(x_seq.detach())  # Detach to prevent backprop through previous steps

            # 2. Calculate Loss against the shifted target
            # Target window slides by 1 at each k
            current_target = seq[:, k + 1 : seq.shape[1] - acc_nsteps + k + 1]
            pred_loss = self.criterion(x_seq, current_target)

            step2loss[f"pred_loss_{k + 1}"] = pred_loss

            reg_loss = self.regularization_loss(x_seq)

            step2loss[f"reg_loss_{k + 1}"] = reg_loss

            if self.hparams.reg_weight is not None and self.hparams.reg_weight > 0.0:
                reg_eff = self.hparams.reg_weight * reg_loss
            else:
                reg_eff = 0.0

            # 3. Accumulate Loss
            if self.hparams.intermediate_losses:
                loss = loss + (pred_loss + reg_eff) / acc_nsteps
            elif k == acc_nsteps - 1:
                loss = loss + (pred_loss + reg_eff)

            # 4. Randomized Detaching (Vectorized)
            # Only apply if not the last step and probability > 0
            if k < acc_nsteps - 1 and self.hparams.detach_probability > 0:
                # Shape: (Batch, Sequence Length) -> Independent prob per sample & time
                # If you want shared across batch like your original code, use size=(1, x_seq.shape[1])
                mask_shape = (x_seq.shape[0], x_seq.shape[1])

                # prob is probability of KEeping the gradient (1 - detach_prob)
                keep_prob = 1.0 - self.hparams.detach_probability

                # Generate mask: 1.0 = Keep Grad, 0.0 = Detach
                mask = torch.bernoulli(torch.full(mask_shape, keep_prob, device=x_seq.device))

                # Reshape mask for broadcasting: (B, L, 1, 1, 1) assuming (B, L, C, H, W)
                # Adds singleton dims for C, H, W
                view_shape = list(mask.shape) + [1] * (x_seq.ndim - 2)
                mask = mask.view(*view_shape)

                # The Mix Trick:
                # When mask is 1: x_seq * 1 + detached * 0 = x_seq (Graph connected)
                # When mask is 0: x_seq * 0 + detached * 1 = detached (Graph cut)
                x_seq = x_seq * mask + x_seq.detach() * (1 - mask)

        if torch.isnan(x_seq).any():
            logger.warning("NaN detected in model predictions!")

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "predictions": x_seq,
            "targets": current_target,  # Return the final target used
        } | step2loss

    def on_train_start(self) -> None:
        """Lightning hook called when training begins."""
        # Reset metrics to avoid pollution from validation sanity checks
        self.valid_loss.reset()
        self.valid_loss_best.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        :param batch: Training batch chunk (b, nsteps, c, *domain_shape)
        :param batch_idx: Batch index
        :return: Training loss
        """
        logger.debug(f"Starting training_step for batch_idx={batch_idx} with batch.shape={batch.shape}")

        seq = batch["seq"]
        step_output = self.model_step(seq)

        # Update metrics
        for metric in self.train_loss:
            self.train_loss[metric](step_output[metric])
            # Log metrics
            self.log(f"train/{metric}", self.train_loss[metric], on_step=False, on_epoch=True)

        return step_output["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx=0) -> dict[str, torch.Tensor] | None:
        if dataloader_idx > 0:
            if batch_idx != 0:
                return
            x_true = batch["seq"]  # (b, nsteps, c, *domain)
            _, nsteps, *_ = x_true.shape
            x = x_true[:, :1].detach()
            with torch.no_grad():
                for _ in range(1, nsteps):
                    x = self.forward(x)
            final_mse = (x[:, 0] - x_true[:, -1]).square().mean()
            if torch.isfinite(final_mse):
                self.log(
                    "valid/final_loss",
                    final_mse,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )
            return

        logger.debug(f"Starting validation_step for batch_idx={batch_idx} with batch.shape={batch.shape}")

        seq = batch["seq"]
        step_output = self.model_step(seq)

        # Update metrics
        for metric in self.valid_loss:
            self.valid_loss[metric](step_output[metric])
            # Log metrics
            add_dataloader_idx = metric != "loss"
            self.log(
                f"valid/{metric}",
                self.valid_loss[metric],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=add_dataloader_idx,
            )

        return {
            "x_seq": seq,
            "x_pre": step_output["predictions"],
        }

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called when validation epoch ends."""
        val_loss = self.valid_loss.compute()
        self.valid_loss_best(val_loss["loss"])
        self.log("valid/loss_best", self.valid_loss_best, sync_dist=True)

    def configure_model(self) -> None:
        if self.hparams.compile and not self._is_compiled:
            self.dynamics = torch.compile(self.dynamics)
            if self.drift is not None:
                self.drift = torch.compile(self.drift)
            self._is_compiled = True

    def setup(self, stage: str) -> None:
        """Lightning hook called at the beginning of fit/validate/test/predict.

        :param stage: Stage name ("fit", "validate", "test", "predict")
        """
        if self.hparams.compile and stage == "fit":
            self.dynamics = torch.compile(self.dynamics)
            if self.drift is not None:
                self.drift = torch.compile(self.drift)
            self._is_compiled = True

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        :return: Dictionary with optimizer and optional scheduler configuration
        """
        # Get parameters from dynamics and drift networks (if drift exists)
        params = list(self.dynamics.parameters())
        if self.drift is not None:
            params.extend(self.drift.parameters())
        optimizer = self.hparams.optimizer(params=params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
