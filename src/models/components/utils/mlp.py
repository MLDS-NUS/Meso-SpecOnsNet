from collections.abc import Sequence

import torch
from torch import nn

from .activation import Activation


class Mlp(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        tgt_dim: int | None = None,
        act_name: str = "relu",
        final_act_name: str | None = None,
        final_bias: bool = True,
        complex: bool = False,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        dtype = torch.complex64 if complex else torch.float32
        for k in range(len(hidden_dims)):
            hdim = hidden_dims[k]
            layers.append(nn.Linear(prev_dim, hdim, dtype=dtype))
            if k == len(hidden_dims) - 1 and final_act_name is not None:  # custom final activation
                layers.append(Activation(final_act_name, split_complex=complex))
            else:
                layers.append(Activation(act_name, split_complex=complex))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, out_dim, dtype=dtype, bias=final_bias))

        self.tgt_dim = tgt_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tgt_dim is None:
            return self.network(x)
        return self.network(x.swapdims(-1, self.tgt_dim)).swapdims(-1, self.tgt_dim)
