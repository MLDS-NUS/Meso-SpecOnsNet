from collections.abc import Sequence

import torch
from torch import nn

from .activation import Activation


class MlpSkipAll(nn.Module):  # deprecated (run 379)
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        tgt_dim: int | None = None,
        act_name: str = "relu",
        require_ln: bool = False,
        final_bias: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(Activation(act_name))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, out_dim, bias=final_bias))
        nn.init.zeros_(layers[-1].bias)
        nn.init.zeros_(layers[-1].weight)

        self.tgt_dim = tgt_dim
        self.require_ln = require_ln
        if require_ln:
            self.ln = nn.LayerNorm(in_dim)
        self.network = nn.Sequential(*layers)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        if self.tgt_dim is not None:
            x = x.swapdims(-1, self.tgt_dim)

        if self.require_ln:
            x = self.ln(x)

        x = self.network(x)

        if self.tgt_dim is not None:
            x = x.swapdims(-1, self.tgt_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inference(x)
