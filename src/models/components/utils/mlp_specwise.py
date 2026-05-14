from collections.abc import Sequence

import einops
import torch
from torch import nn

from .activation import Activation


class SpecLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, Nk: int, norm: str | None = None, bias: bool = True) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Nk = Nk
        self.weight = nn.Parameter(torch.zeros(Nk, in_dim, out_dim))

        if norm == "bn":
            self.norm = nn.BatchNorm1d(in_dim)
        elif norm == "gn" and in_dim % 16 == 0:
            self.norm = nn.GroupNorm(num_groups=16, num_channels=in_dim)
        else:
            self.norm = nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, Nk))
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (bs, in_dim, Nk)
        x = self.norm(x)
        x = einops.einsum(x, self.weight, "... i Nk, Nk i o -> ... o Nk")
        if self.bias is not None:
            x = x + self.bias
        return x


class MlpSpecwise(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        Nk: int,
        norm: str | None = None,
        act_name: str = "relu",
        final_bias: bool = True,
        final_sine: bool = False,
    ):
        super().__init__()

        linears = []
        acts = []
        skips = []
        prev_dim = in_dim
        for k in range(len(hidden_dims)):
            hdim = hidden_dims[k]

            linears.append(SpecLinear(prev_dim, hdim, Nk=Nk, norm=norm))
            if final_sine and k == len(hidden_dims) - 1:
                acts.append(Activation("sine"))
            else:
                acts.append(Activation(act_name))
            skips.append(prev_dim == hdim)
            prev_dim = hdim
        linears.append(SpecLinear(prev_dim, out_dim, Nk=Nk, norm=norm, bias=final_bias))

        self.linears = nn.ModuleList(linears)
        self.acts = nn.ModuleList(acts)
        self.skips = skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (bs, c, Nk=(Nx//2 + 1)*2)

        for k in range(len(self.linears) - 1):
            if self.skips[k]:  # noqa: SIM108
                x = x + self.acts[k](self.linears[k](x))
            else:
                x = self.acts[k](self.linears[k](x))

        x = self.linears[-1](x)

        return x
