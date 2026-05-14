from collections.abc import Sequence

import torch
from torch import nn

from .activation import Activation


class LinearComplex(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        assert in_features % 2 == 0 and out_features % 2 == 0

        self.in_features = in_features
        self.out_features = out_features

        self.real_linear = nn.Linear(in_features // 2, out_features // 2, bias)
        self.imag_linear = nn.Linear(in_features // 2, out_features // 2, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = torch.split(x, self.in_features // 2, dim=-1)

        real = self.real_linear(x_real) - self.imag_linear(x_imag)
        imag = self.real_linear(x_imag) + self.imag_linear(x_real)

        return torch.cat([real, imag], dim=-1)


class MlpSkipComplex(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        tgt_dim: int | None = None,
        act_name: str = "relu",
        final_bias: bool = True,
        final_sine: bool = False,
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
            linears.append(LinearComplex(prev_dim, hdim))
            if final_sine and k == len(hidden_dims) - 1:
                acts.append(Activation("sine"))
            else:
                acts.append(Activation(act_name))
            skips.append(prev_dim == hdim)
            prev_dim = hdim
        linears.append(LinearComplex(prev_dim, out_dim, bias=final_bias))

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
