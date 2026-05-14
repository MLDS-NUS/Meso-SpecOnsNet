import torch
from torch import nn

from .activation import Activation


class SpectralFNO1dBlock(nn.Module):
    def __init__(self, Nx: int, model_dim: int):
        super().__init__()

        self.Nx = Nx
        self.model_dim = model_dim

        self.conv1d_phys = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=1,
        )
        self.conv1d_spec = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=1,
            dtype=torch.cfloat,
        )

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        # x: (bs, model_dim, Nx//2+1) complex-valued
        x_phys = torch.fft.irfft(x_spec, n=self.Nx, dim=-1)  # (bs, model_dim, Nx)
        x_phys = self.conv1d_phys(x_phys)  # (bs, model_dim, Nx)

        x_spec = self.conv1d_spec(x_spec)  # (bs, model_dim, Nx//2+1)

        return x_spec + torch.fft.rfft(x_phys, dim=-1)


class SpectralFNO1d(nn.Module):
    def __init__(
        self,
        Nx: int,
        ndim: int,
        model_dim: int,
        n_blocks: int,
        act_name: str = "relu",
        final_bias: bool = True,
        final_sine: bool = False,
    ):
        super().__init__()

        self.Nx = Nx
        self.ndim = ndim
        self.model_dim = model_dim
        self.n_blocks = n_blocks

        self.lift = nn.Conv1d(ndim, model_dim, kernel_size=1, dtype=torch.cfloat)
        self.proj = nn.Conv1d(model_dim, ndim, kernel_size=1, bias=final_bias, dtype=torch.cfloat)
        self.blocks = nn.ModuleList([SpectralFNO1dBlock(Nx=Nx, model_dim=model_dim) for _ in range(n_blocks)])
        self.act = Activation(act_name)
        self.act_last = Activation("sine") if final_sine else Activation(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, ndim, Nx//2+1) complex-valued
        x = self.lift(x)  # (bs, model_dim, Nx//2+1)

        for k in range(self.n_blocks):
            x = self.blocks[k](x)
            x = torch.fft.irfft(x, n=self.Nx, dim=-1)
            x = self.act_last(x) if k == self.n_blocks - 1 else self.act(x)
            x = torch.fft.rfft(x, dim=-1)  # (bs, model_dim, Nx//2+1)

        x = self.proj(x)  # (bs, ndim, Nx//2+1)

        return x
