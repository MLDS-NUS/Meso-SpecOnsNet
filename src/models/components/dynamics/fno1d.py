import torch
from neuralop.models import FNO

from .dynamics_base import DynamicsBase


class FNO1dBaseline(DynamicsBase):
    def __init__(
        self,
        n_modes: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        step_method: str = "euler",
    ):
        super().__init__(step_method=step_method)

        self.fno1d = FNO(
            n_modes=(n_modes,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def L(self, u: torch.Tensor) -> torch.Tensor:
        return self.fno1d(u)
