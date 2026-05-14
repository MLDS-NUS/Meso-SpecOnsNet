from collections.abc import Sequence

import torch
from torch import nn

from .activation import Activation


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding_mode: str = "circular",
        act_name: str = "relu",
        final_bias: bool = True,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
            bias=final_bias,
        )
        self.activation = Activation(act_name)

        self.skip_connection = skip_connection and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv1(x)
        z = self.activation(z)
        z = self.conv2(z)
        if self.skip_connection:
            return z + x
        else:
            return z


class CNN1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        kernel_size: int,
        padding_mode: str = "circular",
        act_name: str = "relu",
        final_bias: bool = True,
        inner_final_bias: bool = True,
        skip_connection: bool = False,
    ):
        super().__init__()
        layers = []

        # conv-in
        layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            )
        )
        layers.append(Activation(act_name))

        # conv-hidden
        for in_c, out_c in zip(hidden_channels[:-1], hidden_channels[1:]):
            layers.append(
                Conv1dBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    act_name=act_name,
                    final_bias=inner_final_bias,
                    skip_connection=skip_connection,
                )
            )
            layers.append(Activation(act_name))

        # conv-out
        layers.append(
            nn.Conv1d(
                in_channels=hidden_channels[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
                bias=final_bias,
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
