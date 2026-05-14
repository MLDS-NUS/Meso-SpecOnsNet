from collections.abc import Sequence

import torch
from torch import nn

from .activation import Activation


class MlpSkip(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        tgt_dim: int | None = None,
        act_name: str = "relu",
        final_bias: bool = True,
        final_sine: bool = False,
        final_zero: bool = False,
    ):
        super().__init__()
        # LayerNorm for all but the last layer does not improve (run 377)

        # lns = []
        linears = []
        acts = []
        skips = []
        prev_dim = in_dim
        for k in range(len(hidden_dims)):
            hdim = hidden_dims[k]
            # lns.append(nn.LayerNorm(prev_dim))
            linears.append(nn.Linear(prev_dim, hdim))
            if final_sine and k == len(hidden_dims) - 1:
                acts.append(Activation("sine"))
            else:
                acts.append(Activation(act_name))
            skips.append(prev_dim == hdim)
            prev_dim = hdim
        linears.append(nn.Linear(prev_dim, out_dim, bias=final_bias))
        if final_zero:
            nn.init.zeros_(linears[-1].weight)
            if final_bias:
                nn.init.zeros_(linears[-1].bias)

        self.tgt_dim = tgt_dim
        # self.lns = nn.ModuleList(lns)
        self.linears = nn.ModuleList(linears)
        self.acts = nn.ModuleList(acts)
        self.skips = skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tgt_dim is not None:
            x = x.swapdims(-1, self.tgt_dim)

        for k in range(len(self.linears) - 1):
            x_in = x
            # x = self.lns[k](x)
            x = self.linears[k](x)
            x = self.acts[k](x)
            if self.skips[k]:
                x = x + x_in  # skip connection

        x = self.linears[-1](x)

        if self.tgt_dim is not None:
            x = x.swapdims(-1, self.tgt_dim)

        return x
