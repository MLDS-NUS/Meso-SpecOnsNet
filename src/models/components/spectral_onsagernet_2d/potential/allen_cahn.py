import torch
from torch import nn

from .potential_base import Potential


class AllenCahnPotential(Potential):
    r"""Analytical Allen-Cahn free energy in 2D.

    V[u] = ∫ 0.5 * a * u² + 0.5 * b * |∇u|² + W(u) dx dy
    W(u) = u⁴/(4ε²) - u²/(2ε²),  W'(u) = (u³ - u)/ε²
    Gradient flow: u_t = -δV/δu = bΔu - au + (u - u³)/ε² (with a=0, b=1).
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        eps: float = 0.1,
    ):
        super().__init__()

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        self.eps = eps

        self.dummy_param = nn.Parameter(torch.tensor(0.0))

        # rfft2 output shape on (..., Nx, Ny): (..., Nx, Ny // 2 + 1).
        # x dim is full FFT (fftfreq); y dim is rfft (rfftfreq).
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dx)  # (Nx,)
        freq_y = torch.fft.rfftfreq(self.Ny, d=self.dy)  # (Ny // 2 + 1,)
        kx = 2.0 * torch.pi * freq_x
        ky = 2.0 * torch.pi * freq_y
        self.register_buffer("kx", kx.view(-1, 1))  # (Nx, 1)
        self.register_buffer("ky", ky.view(1, -1))  # (1, Ny // 2 + 1)
        self.register_buffer("k_sq", self.kx**2 + self.ky**2)  # (Nx, Ny // 2 + 1)

        self.kx: torch.Tensor
        self.ky: torch.Tensor
        self.k_sq: torch.Tensor

    @property
    def a(self):
        return self.dummy_param * 0.0

    @property
    def b(self):
        return 1.0

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """V0 = ∫ 0.5 * (a u² + b |∇u|²) dx dy."""
        u_h = torch.fft.rfft2(u)
        grad_x = torch.fft.irfft2(1j * self.kx * u_h, s=(self.Nx, self.Ny))
        grad_y = torch.fft.irfft2(1j * self.ky * u_h, s=(self.Nx, self.Ny))

        integrand = 0.5 * self.a * u**2 + 0.5 * self.b * (grad_x**2 + grad_y**2)
        Vu = torch.sum(integrand, dim=(-2, -1)) * self.dx * self.dy  # (..., c)
        return Vu.sum(dim=-1, keepdim=True)  # (..., 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """V1 = ∫ W(u) dx dy."""
        Fu = self.F(u)
        Vu = torch.sum(Fu, dim=(-2, -1)) * self.dx * self.dy  # (..., c)
        return Vu.sum(dim=-1, keepdim=True)  # (..., 1)

    def V(self, u: torch.Tensor) -> torch.Tensor:
        return self._V0(u) + self._V1(u)

    def _dV0du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        return (self.a + self.b * self.k_sq) * u_hat

    def _dV1du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        u = torch.fft.irfft2(u_hat, s=(self.Nx, self.Ny))
        Fpu = self.Fp(u)
        return torch.fft.rfft2(Fpu)

    def dVdu_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        return self.L_hat() * u_hat + self._dV1du_h(u_hat)

    def dVdu(self, u: torch.Tensor) -> torch.Tensor:
        u_hat = torch.fft.rfft2(u)
        dVdu_h = self.dVdu_h(u_hat)
        return torch.fft.irfft2(dVdu_h, s=(self.Nx, self.Ny))

    def F(self, u: torch.Tensor) -> torch.Tensor:
        """Double-well density W(u) = u⁴/(4ε²) - u²/(2ε²)."""
        return u**4 / (4 * self.eps**2) - u**2 / (2 * self.eps**2)

    def Fp(self, u: torch.Tensor) -> torch.Tensor:
        """W'(u) = (u³ - u)/ε²."""
        return (u**3 - u) / self.eps**2

    def L_hat(self) -> torch.Tensor:
        """Linear part of dVdu_h (matches _dV0du_h), used by SBDF1."""
        return self.a + self.b * self.k_sq

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Nonlinear part of dVdu_h (matches _dV1du_h), used by SBDF1."""
        return self._dV1du_h(u_hat)
