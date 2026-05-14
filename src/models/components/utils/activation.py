import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self, act_name: str | tuple[str, dict], split_complex: bool = False) -> None:
        super().__init__()

        if isinstance(act_name, tuple):
            act_name, act_kwargs = act_name
        else:
            act_kwargs = {}

        if act_name == "relu":
            self.act = nn.ReLU(**act_kwargs)
        elif act_name == "gelu":
            self.act = nn.GELU(**act_kwargs)
        elif act_name in ["swish", "silu"]:
            self.act = nn.SiLU(**act_kwargs)
        elif act_name == "tanh":
            self.act = nn.Tanh()
        elif act_name == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_name in ["leaky", "leaky_relu"]:
            self.act = nn.LeakyReLU(**act_kwargs)
        elif act_name == "softplus":
            self.act = nn.Softplus(**act_kwargs)
        elif act_name == "sine":
            self.act = lambda x: torch.sin(x)
        else:
            raise ValueError(f"Unsupported activation: {act_name}")

        self.split_complex = split_complex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split_complex:
            return torch.complex(self.act(x.real), self.act(x.imag))
        else:
            return self.act(x)
