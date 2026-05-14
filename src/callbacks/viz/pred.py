"""Visualization callback for 1D prediction as space-time image."""

import einops
import lightning as L
import numpy as np
import tensordict as td
import torch
from matplotlib import pyplot as plt

from src.models.meso_module import MesoLitModule
from src.utils.pylogger import RankedLogger

from ..utils import log_figure

log = RankedLogger(__name__, rank_zero_only=True)


class PredViz(L.Callback):
    """Visualization callback for plotting 1D predictions as space-time images.

    Plots predictions where:
    - x-axis: spatial dimension
    - y-axis: time index
    - color: field value
    """

    def __init__(
        self,
        n_samples: int = 4,
        max_timesteps: int = None,
        every_n_validation: int = 1,
    ) -> None:
        """Initialize the prediction visualization callback.

        Args:
            n_samples: Number of samples to visualize from validation set
            max_timesteps: Maximum number of timesteps to plot (None = all)
            every_n_validation: Plot every N validation epochs (default: 1 = every epoch)
        """
        super().__init__()
        self.n_samples = n_samples
        self.max_timesteps = max_timesteps
        self.every_n_validation = every_n_validation

        # Storage for predictions
        self.x_trues = []
        self.x_preds = []
        self.valid_steps = []  # Store valid timesteps for each sample
        self.collected_samples = 0
        self.validation_epoch_count = 0

    def _compute_rollout(
        self,
        pl_module: MesoLitModule,
        x_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rollout prediction for full trajectories, stopping at NaN.

        Args:
            pl_module: Lightning module
            x_true: Ground truth sequence (b, T, c, h, w) or (b, T, c, w) for 1D

        Returns:
            Tuple of (predicted sequence, valid_steps) where:
            - predicted sequence: (b, T, c, h, w) or (b, T, c, w)
            - valid_steps: (b,) tensor indicating last valid timestep for each sample
        """
        b, nsteps = x_true.shape[0], x_true.shape[1]
        x = x_true[:, 0]

        x_pred = torch.empty_like(x_true)
        x_pred[:, 0] = x.clone()

        # Track valid steps for each sample in batch
        valid_steps = torch.full((b,), nsteps - 1, dtype=torch.long, device=x.device)
        batch_active = torch.ones(b, dtype=torch.bool, device=x.device)

        for step in range(1, nsteps):
            # Predict next state - add sequence dimension
            pred = pl_module.forward(x.unsqueeze(1)).detach().clone()
            pred = pred.squeeze(1)  # Remove sequence dimension
            pred.requires_grad_(False)

            # Check for NaN in each sample
            has_nan = torch.isnan(pred).any(dim=tuple(range(1, pred.ndim)))  # (b,)

            # Mark samples with NaN as inactive
            batch_active = batch_active & ~has_nan

            # Update valid_steps for samples that just became invalid
            newly_invalid = has_nan & (valid_steps == nsteps - 1)
            valid_steps[newly_invalid] = step - 1

            # Store predictions (NaN will be included but we track valid_steps)
            x_pred[:, step] = pred.clone()

            # If all samples have NaN, stop early
            if not batch_active.any():
                log.info(f"All samples have NaN at step {step}, stopping rollout early")
                break

            # Update for next prediction (use last valid prediction for inactive samples)
            x = pred.detach().clone()
            x.requires_grad_(False)

        return x_pred, valid_steps

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: MesoLitModule,
    ) -> None:
        """Reset storage at the start of validation epoch."""
        self.validation_epoch_count += 1

        # Only reset storage if we're plotting this epoch
        if self.validation_epoch_count % self.every_n_validation == 0:
            self.x_trues = []
            self.x_preds = []
            self.valid_steps = []
            self.collected_samples = 0

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: MesoLitModule,
        outputs: dict[str, torch.Tensor] | None,
        batch: td.TensorDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions from validation batches.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            outputs: Model outputs
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index (0 for 'valid', 1 for 'valid_full')
        """
        # Skip if not plotting this epoch
        if self.validation_epoch_count % self.every_n_validation != 0:
            return

        # Only use valid_full dataloader
        if dataloader_idx != 1:
            return

        if self.collected_samples >= self.n_samples:
            return

        samples_needed = self.n_samples - self.collected_samples
        samples_to_take = min(samples_needed, batch["seq"].shape[0])

        x_true = batch["seq"][:samples_to_take]  # (b, T, c, ...)
        x_pred, valid_steps = self._compute_rollout(pl_module, x_true)

        self.x_trues.append(einops.asnumpy(x_true))
        self.x_preds.append(einops.asnumpy(x_pred))
        self.valid_steps.append(einops.asnumpy(valid_steps))
        self.collected_samples += samples_to_take

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: MesoLitModule,
    ) -> None:
        """Plot space-time images at end of validation epoch."""
        # Skip if not plotting this epoch
        if self.validation_epoch_count % self.every_n_validation != 0:
            return

        if len(self.x_trues) == 0:
            return

        self._plot_spacetime_images(trainer=trainer)

    def _plot_spacetime_images(self, trainer: L.Trainer) -> None:
        """Plot space-time images for collected samples.

        Args:
            trainer: Lightning trainer
        """
        # Concatenate all batches
        x_true = np.concatenate(self.x_trues, axis=0)  # (n_samples, T, c, ...)
        x_pred = np.concatenate(self.x_preds, axis=0)  # (n_samples, T, c, ...)
        valid_steps = np.concatenate(self.valid_steps, axis=0)  # (n_samples,)

        n_samples, n_timesteps, n_channels = x_true.shape[:3]

        # Check if data is 1D or 2D
        is_1d = len(x_true.shape) == 4  # (n_samples, T, c, w)

        # Create a single figure with all samples
        # Layout: rows = 3 * n_channels (truth, pred, error per channel)
        #         cols = n_samples
        n_rows = 3 * n_channels
        fig, axs = plt.subplots(n_rows, n_samples, figsize=(n_samples * 4, n_rows * 3), squeeze=False)

        for sample_idx in range(n_samples):
            x_true_sample = x_true[sample_idx]  # (T, c, ...)
            x_pred_sample = x_pred[sample_idx]  # (T, c, ...)
            valid_t = int(valid_steps[sample_idx]) + 1  # +1 to include the last valid step

            # Limit to valid timesteps and max_timesteps
            if self.max_timesteps is not None:
                valid_t = min(valid_t, self.max_timesteps)

            # Truncate to valid steps
            x_true_sample = x_true_sample[:valid_t]
            x_pred_sample = x_pred_sample[:valid_t]

            for c_idx in range(n_channels):
                if is_1d:
                    # For 1D: shape is (T, w)
                    true_data = x_true_sample[:, c_idx, :]  # (T, w)
                    pred_data = x_pred_sample[:, c_idx, :]  # (T, w)
                else:
                    # For 2D: shape is (T, h, w) - take middle slice
                    h_mid = x_true_sample.shape[2] // 2
                    true_data = x_true_sample[:, c_idx, h_mid, :]  # (T, w)
                    pred_data = x_pred_sample[:, c_idx, h_mid, :]  # (T, w)

                error_data = pred_data - true_data

                # Compute shared colorbar limits based on initial condition with ±10% margin
                initial_true_data = x_true_sample[0, c_idx, :] if is_1d else x_true_sample[0, c_idx, h_mid, :]
                data_min = initial_true_data.min()
                data_max = initial_true_data.max()
                data_range = data_max - data_min
                vmin = data_min - 0.1 * data_range
                vmax = data_max + 0.1 * data_range

                # Row indices for this channel
                truth_row = 3 * c_idx
                pred_row = 3 * c_idx + 1
                error_row = 3 * c_idx + 2

                # Plot ground truth
                im_true = axs[truth_row, sample_idx].imshow(
                    true_data,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
                axs[truth_row, sample_idx].set_xlabel("Spatial index", fontsize=8)
                if sample_idx == 0:
                    axs[truth_row, sample_idx].set_ylabel(f"Time\nCh{c_idx} Truth", fontsize=9, fontweight="bold")
                if truth_row == 0:
                    axs[truth_row, sample_idx].set_title(
                        f"Sample {sample_idx}\n(Valid: {valid_t})", fontsize=10, fontweight="bold"
                    )
                axs[truth_row, sample_idx].tick_params(labelsize=7)
                plt.colorbar(im_true, ax=axs[truth_row, sample_idx], fraction=0.046, pad=0.04)

                # Plot prediction
                im_pred = axs[pred_row, sample_idx].imshow(
                    pred_data,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
                axs[pred_row, sample_idx].set_xlabel("Spatial index", fontsize=8)
                if sample_idx == 0:
                    axs[pred_row, sample_idx].set_ylabel(f"Time\nCh{c_idx} Pred", fontsize=9, fontweight="bold")
                axs[pred_row, sample_idx].tick_params(labelsize=7)
                plt.colorbar(im_pred, ax=axs[pred_row, sample_idx], fraction=0.046, pad=0.04)

                # Plot error
                error_abs_max = np.abs(error_data).max()
                im_error = axs[error_row, sample_idx].imshow(
                    error_data,
                    aspect="auto",
                    origin="lower",
                    vmin=-error_abs_max,
                    vmax=error_abs_max,
                    cmap="bwr",
                )
                axs[error_row, sample_idx].set_xlabel("Spatial index", fontsize=8)
                if sample_idx == 0:
                    axs[error_row, sample_idx].set_ylabel(f"Time\nCh{c_idx} Error", fontsize=9, fontweight="bold")

                # Compute error metrics
                mse = np.mean(error_data**2)
                mae = np.mean(np.abs(error_data))
                axs[error_row, sample_idx].text(
                    0.5,
                    -0.15,
                    f"MSE={mse:.2e}\nMAE={mae:.2e}",
                    ha="center",
                    va="top",
                    transform=axs[error_row, sample_idx].transAxes,
                    fontsize=7,
                )
                axs[error_row, sample_idx].tick_params(labelsize=7)
                plt.colorbar(im_error, ax=axs[error_row, sample_idx], fraction=0.046, pad=0.04)

        plt.tight_layout()

        log_figure(
            trainer=trainer,
            fig=fig,
            key=f"valid/{self.__class__.__name__}/spacetime",
        )
        plt.close(fig)
