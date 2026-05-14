"""Evaluation script for mesoscopic dynamics models.

This script loads a trained checkpoint, runs evaluation on validation/test data,
summarizes metrics, and generates visualizations via callbacks.
"""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
console = Console()


def load_checkpoint_config(ckpt_path: str | Path) -> DictConfig | None:
    """Load hyperparameters from checkpoint if available.

    :param ckpt_path: Path to the checkpoint file.
    :return: DictConfig with hyperparameters or None if not available.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        log.warning(f"Checkpoint not found: {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" in checkpoint:
        log.info("Found hyperparameters in checkpoint")
        return OmegaConf.create(checkpoint["hyper_parameters"])

    return None


def print_metrics_table(metrics: dict[str, Any], title: str = "Evaluation Metrics") -> None:
    """Print metrics in a formatted Rich table.

    :param metrics: Dictionary of metric names to values.
    :param title: Title for the table.
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Value", justify="right", style="green", width=20)

    for name, value in sorted(metrics.items()):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, float):
            table.add_row(name, f"{value:.6f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def compute_additional_metrics(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
) -> dict[str, float]:
    """Compute additional evaluation metrics beyond standard validation.

    :param trainer: Lightning trainer.
    :param model: Lightning module.
    :param datamodule: Lightning data module.
    :return: Dictionary of additional metrics.
    """
    additional_metrics = {}

    # Get validation dataloader
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    # Handle multiple dataloaders
    if isinstance(val_loader, list):
        val_loader = val_loader[0]

    model.eval()
    device = next(model.parameters()).device

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device) if isinstance(batch, dict) else batch.to(device)

            # Run model step
            step_output = model.model_step(seq)

            predictions = step_output["predictions"]
            targets = step_output["targets"]

            # Compute MSE and MAE
            mse = torch.nn.functional.mse_loss(predictions, targets, reduction="sum")
            mae = torch.nn.functional.l1_loss(predictions, targets, reduction="sum")

            batch_size = predictions.shape[0]
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += batch_size * predictions.numel() // batch_size

    if total_samples > 0:
        additional_metrics["mse_per_element"] = total_mse / total_samples
        additional_metrics["mae_per_element"] = total_mae / total_samples
        additional_metrics["rmse_per_element"] = (total_mse / total_samples) ** 0.5

    return additional_metrics


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a trained model checkpoint.

    This function:
    1. Loads the checkpoint and corresponding model
    2. Runs validation to compute metrics
    3. Executes visualization callbacks
    4. Prints a summary of all metrics

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path, "Checkpoint path (ckpt_path) is required for evaluation!"

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint from: {ckpt_path}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks for visualization
    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate loggers
    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Run validation with checkpoint
    log.info("Starting validation evaluation...")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Collect validation metrics
    metric_dict = dict(trainer.callback_metrics)

    # Compute additional metrics
    log.info("Computing additional metrics...")
    additional_metrics = compute_additional_metrics(trainer, model, datamodule)
    metric_dict.update(additional_metrics)

    # Print metrics summary
    console.print("\n")
    print_metrics_table(metric_dict, title="Evaluation Results")

    # Optionally run test if test dataloader exists
    if cfg.get("run_test", False):
        log.info("Running test evaluation...")
        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
            test_metrics = {f"test/{k}": v for k, v in trainer.callback_metrics.items() if "test" in k}
            if test_metrics:
                print_metrics_table(test_metrics, title="Test Results")
                metric_dict.update(test_metrics)
        except Exception as e:
            log.warning(f"Test evaluation failed: {e}")

    # Save metrics to file if output path specified
    if cfg.get("output_path"):
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to floats for serialization
        serializable_metrics = {}
        for k, v in metric_dict.items():
            if isinstance(v, torch.Tensor):
                serializable_metrics[k] = v.item()
            else:
                serializable_metrics[k] = v

        import json

        with open(output_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        log.info(f"Metrics saved to: {output_path}")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Register eval resolver for dynamic config values
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)

    # Apply extra utilities
    extras(cfg)

    # Run evaluation
    metric_dict, _ = evaluate(cfg)

    # Print final summary
    console.print("\n[bold green]Evaluation completed successfully![/bold green]")

    # Return best metric if specified
    if cfg.get("optimized_metric"):
        metric_name = cfg.optimized_metric
        if metric_name in metric_dict:
            console.print(f"[bold]Optimized metric ({metric_name}): {metric_dict[metric_name]:.6f}[/bold]")


if __name__ == "__main__":
    main()
