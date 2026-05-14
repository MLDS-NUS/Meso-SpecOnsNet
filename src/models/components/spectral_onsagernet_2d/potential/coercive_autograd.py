import torch
from torch import nn
from torch.nn import functional as F

from .potential_base import Potential


# migrated from coercive_autograd_v6.py for the 1d case
class CoerciveAutogradPotential(Potential):
    def __init__(
        self,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        F_module: nn.Module,
        v_module: nn.Module,
        v_truncate: int,
    ):
        super().__init__()

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        self.a_params = nn.Parameter(torch.zeros(1))
        self.b_params = nn.Parameter(torch.zeros(1))

        self.F_module = F_module
        self.v_module = v_module

        self.v_truncate = v_truncate

        # rfft2 output shape: (..., Nx, Ny // 2 + 1)
        # x dim is full FFT (fftfreq), y dim is rfft (rfftfreq).
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dx)  # (Nx,)
        freq_y = torch.fft.rfftfreq(self.Ny, d=self.dy)  # (Ny // 2 + 1,)
        self.register_buffer("freq_x", freq_x.view(-1, 1))  # (Nx, 1)
        self.register_buffer("freq_y", freq_y.view(1, -1))  # (1, Ny // 2 + 1)
        self.freq_x: torch.Tensor
        self.freq_y: torch.Tensor

    @property
    def a(self):
        return F.softplus(self.a_params)

    @property
    def b(self):
        return F.softplus(self.b_params)

    def Fmap(self, u: torch.Tensor) -> torch.Tensor:
        """Pointwise Nonlinear Potential Energy Density F(u)."""
        # u.shape: (bs, c, Nx, Ny)
        return self.F_module(u.swapaxes(1, -1)).swapaxes(1, -1)

    def vmap(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Nonlocal Potential Energy Density V(u)."""
        # u_hat: (bs, c, Nx, Ny // 2 + 1) -> (bs, c, 1)
        truncated = u_hat[..., : self.v_truncate, : self.v_truncate]
        flat = truncated.flatten(-2)  # (bs, c, v_truncate * v_truncate)
        vu = self.v_module(flat)
        return vu.abs()

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """pointwise information
        V0 = ∫ 0.5 * a * |u|^2 + F(u(x, y))) dx dy
        """
        integrand = 0.5 * self.a * u**2 + self.Fmap(u)  # (bs, c, Nx, Ny)
        Vu = torch.sum(integrand, dim=(-2, -1)) * self.dx * self.dy  # (bs, c)
        return Vu.sum(dim=-1, keepdim=True)  # (bs, 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """local information
        V1 = ∫ 0.5 * b * (|u_x|^2 + |u_y|^2) dx dy
        """
        u_h = torch.fft.rfft2(u)

        grad_x_h = 1j * 2.0 * torch.pi * self.freq_x * u_h
        grad_y_h = 1j * 2.0 * torch.pi * self.freq_y * u_h
        grad_x = torch.fft.irfft2(grad_x_h, s=(self.Nx, self.Ny))
        grad_y = torch.fft.irfft2(grad_y_h, s=(self.Nx, self.Ny))

        integrand = 0.5 * self.b * (grad_x**2 + grad_y**2)  # (bs, c, Nx, Ny)
        Vu = torch.sum(integrand, dim=(-2, -1)) * self.dx * self.dy  # (bs, c)

        return Vu.sum(dim=-1, keepdim=True)  # (bs, 1)

    def _V2(self, u: torch.Tensor) -> torch.Tensor:
        """nonlocal information"""
        u_hat = torch.fft.rfft2(u)
        Vu = self.vmap(u_hat)  # (bs, c, Nx, Ny) -> (bs, 1)
        return Vu

    def V(self, u: torch.Tensor) -> torch.Tensor:
        """Total Energy for plotting."""
        return self._V0(u) + self._V1(u) + self._V2(u)

    def _dFdu(self, u: torch.Tensor) -> torch.Tensor:
        """dF/du using autograd."""
        with torch.inference_mode(False):
            u.requires_grad_(True)
            Fu = self.Fmap(u)
            dFdu = torch.autograd.grad(
                outputs=Fu.sum(),
                inputs=u,
                create_graph=True,
                retain_graph=True,
            )[0]
        return dFdu

    def _dV0du(self, u: torch.Tensor) -> torch.Tensor:
        """dV0/du = au + F'(u)"""
        au = self.a * u
        dFdu = self._dFdu(u)
        return au + dFdu

    def _dV0du_h(self, u_h: torch.Tensor) -> torch.Tensor:
        """dV0/du"""
        u = torch.fft.irfft2(u_h, s=(self.Nx, self.Ny))
        dV0du = self._dV0du(u)
        dV0du_h = torch.fft.rfft2(dV0du)
        return dV0du_h

    def _dV1du_h(self, u_h: torch.Tensor) -> torch.Tensor:
        """dV1/du = -bΔu"""
        return self.b * (2 * torch.pi) ** 2 * (self.freq_x**2 + self.freq_y**2) * u_h

    def _dV2du_h(self, u_h: torch.Tensor) -> torch.Tensor:
        """dV2/du"""
        with torch.inference_mode(False):
            u_h.requires_grad_(True)
            vu = self.vmap(u_h)
            dVdu_h = torch.autograd.grad(
                outputs=vu.sum(),
                inputs=u_h,
                create_graph=True,
                retain_graph=True,
            )[0]
        return dVdu_h

    def dVdu_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Total Spectral Force."""
        return self._dV0du_h(u_hat) + self._dV1du_h(u_hat) + self._dV2du_h(u_hat)

    def dVdu(self, u: torch.Tensor) -> torch.Tensor:
        """Physical Force (for plotting)."""
        u_hat = torch.fft.rfft2(u)
        dVdu_h = self.dVdu_h(u_hat)
        return torch.fft.irfft2(dVdu_h, s=(self.Nx, self.Ny))

    def L_hat(self) -> torch.Tensor:
        """
        Linear operator for Implicit SBDF1.
        Must match the linear part dV0du_h + dV1du_h.
        """
        return self.a + self.b * (2 * torch.pi) ** 2 * (self.freq_x**2 + self.freq_y**2)

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Non-linear part for Explicit SBDF1.
        """
        return self.dVdu_h(u_hat) - self.L_hat() * u_hat
