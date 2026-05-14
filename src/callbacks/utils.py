import lightning as L
from lightning.pytorch import loggers
from matplotlib import pyplot as plt
from PIL import Image


def _fig2img(fig: plt.Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    # Save figure to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    # Open with PIL
    img = Image.open(buf)
    return img.copy()  # Copy to avoid issues when buffer is closed


def log_figure(
    trainer: L.Trainer,
    fig: plt.Figure | Image.Image,
    key: str | None = None,
    artifact_file: str | None = None,
) -> None:
    for logger in trainer.loggers:
        if isinstance(logger, loggers.WandbLogger) and key is not None:
            img = _fig2img(fig) if isinstance(fig, plt.Figure) else fig
            logger.log_image(key=key, images=[img], step=trainer.global_step)

        # elif isinstance(logger, loggers.MLFlowLogger) and artifact_file is not None:
        #     # https://github.com/Lightning-AI/pytorch-lightning/issues/3964#issuecomment-705348121
        #     # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html?highlight=client#mlflow.client.MlflowClient.log_figure
        #     logger.experiment.log_figure(
        #         run_id=logger.run_id,
        #         figure=fig,
        #         artifact_file=artifact_file,
        #         # step=trainer.global_step,
        #     )

        elif isinstance(logger, loggers.TensorBoardLogger):
            # do whatever the tensorboard logger supports
            pass
