import torch
from torch import nn

from .potential_base import Potential


class KdVPotential(Potential):
    def __init__(
        self,
        Lx: float,
        Nx: int,
    ):
        super().__init__()

        self.Lx = Lx
        self.Nx = Nx
        self.dx = Lx / Nx

        self.dummy_param = nn.Parameter(torch.tensor(0.0))

        # 1. Precompute Wave Numbers (k)
        # 2*pi*k/L is the physical wavenumber.
        k_freq = torch.fft.rfftfreq(self.Nx, d=self.dx)
        k = 2.0 * torch.pi * k_freq
        self.register_buffer("freq", k)
        self.register_buffer("k_sq", k**2)

        self.freq: torch.Tensor
        self.k_sq: torch.Tensor

    @property
    def a(self):
        return self.dummy_param * 0.0

    @property
    def b(self):
        return 1.0

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """
        V0 = ∫ 0.5 * ( a*u^2 + b*(du/dx)^2 ) dx
        """
        u_h = torch.fft.rfft(u, dim=-1)

        grad_u = torch.fft.irfft(1j * self.freq * u_h, n=self.Nx, dim=-1)

        integrand = 0.5 * self.a * u**2 + 0.5 * self.b * grad_u**2

        Vu = torch.sum(integrand, dim=-1) * self.dx  # (B,)
        return Vu.unsqueeze(-1)  # (B, 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear Potential Energy in Physical Space.
        V1 = ∫ ( -u^3 ) dx
        """
        integrand = -(u**3)
        Vu = torch.sum(integrand, dim=-1) * self.dx  # (B,)
        return Vu.unsqueeze(-1)  # (B, 1)

    def V(self, u: torch.Tensor) -> torch.Tensor:
        """Total Energy for plotting."""
        return self._V0(u) + self._V1(u)

    def _dV0du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        return (self.a + self.b * self.k_sq) * u_hat

    def _dV1du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear Potential Gradient in Spectral Space.
        dV1/du = -3u^2
        """
        u = torch.fft.irfft(u_hat, n=self.Nx, dim=-1)
        grad_phys = -3 * (u**2)
        grad_hat = torch.fft.rfft(grad_phys, dim=-1)
        return grad_hat

    def dVdu_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        return self.L_hat() * u_hat + self._dV1du_h(u_hat)

    def dVdu(self, u: torch.Tensor) -> torch.Tensor:
        u_hat = torch.fft.rfft(u, dim=-1)
        dVdu_h = self.dVdu_h(u_hat)
        return torch.fft.irfft(dVdu_h, n=self.Nx, dim=-1)

    def F(self, u: torch.Tensor) -> torch.Tensor:
        return -(u**3)

    def Fp(self, u: torch.Tensor) -> torch.Tensor:
        return -3 * u**2

    def L_hat(self) -> torch.Tensor:
        """
        Linear operator for Implicit SBDF1.
        Must match the linear part dV0du_h.
        """
        return self.a + self.b * self.k_sq

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Non-linear part for Explicit SBDF1.
        Must match the nonlinear part dV1du_h.
        """
        return self._dV1du_h(u_hat)
