import torch
from torch import nn
from torch.nn import functional as F

from .potential_base import Potential


class CoerciveAutogradPotentialV6(Potential):
    def __init__(
        self,
        Lx: float,
        Nx: int,
        F_module: nn.Module,
        v_module: nn.Module,
        v_truncate: int,
    ):
        super().__init__()

        self.Lx = Lx
        self.Nx = Nx
        self.dx = Lx / Nx

        self.a_params = nn.Parameter(torch.zeros(1))
        self.b_params = nn.Parameter(torch.zeros(1))

        self.F_module = F_module
        self.v_module = v_module

        self.v_truncate = v_truncate

        self.register_buffer("freq", torch.fft.rfftfreq(self.Nx, d=self.dx))
        self.freq: torch.Tensor

    @property
    def a(self):
        return F.softplus(self.a_params)

    @property
    def b(self):
        return F.softplus(self.b_params)

    def Fmap(self, u: torch.Tensor) -> torch.Tensor:
        """Pointwise Nonlinear Potential Energy Density F(u)."""
        # u.shape: (bs, c, Nx)
        return self.F_module(u.swapaxes(1, -1)).swapaxes(1, -1)

    def vmap(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Nonlocal Potential Energy Density V(u)."""
        # (bs, c, Nx) -> (bs, 1)
        vu = self.v_module(u_hat[..., : self.v_truncate])
        return vu.abs()

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """pointwise information
        V0 = ∫ 0.5 * a * |u|^2 + F(u(x))) dx
        """
        integrand = 0.5 * self.a * u**2 + self.Fmap(u)  # (bs, c, Nx)
        Vu = torch.sum(integrand, dim=-1) * self.dx  # (bs, c)
        return Vu.sum(dim=-1, keepdim=True)  # (bs, 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """local information
        V0 = ∫ 0.5 * b * |u_x|^2 dx
        """
        u_h = torch.fft.rfft(u)
        grad_u = torch.fft.irfft(1j * 2.0 * torch.pi * self.freq * u_h)
        integrand = 0.5 * self.b * grad_u**2  # (bs, c, Nx)
        Vu = torch.sum(integrand, dim=-1) * self.dx  # (bs, c)
        return Vu.sum(dim=-1, keepdim=True)  # (bs, 1)

    def _V2(self, u: torch.Tensor) -> torch.Tensor:
        """nonlocal information"""
        u_hat = torch.fft.rfft(u)
        Vu = self.vmap(u_hat)  # (bs, c, Nx) -> (bs, 1)
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
        u = torch.fft.irfft(u_h)
        dV0du = self._dV0du(u)
        dV0du_h = torch.fft.rfft(dV0du)
        return dV0du_h

    def _dV1du_h(self, u_h: torch.Tensor) -> torch.Tensor:
        """dV1/du = -bΔu"""
        return self.b * (2 * torch.pi) ** 2 * (self.freq) ** 2 * u_h

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
        u_hat = torch.fft.rfft(u)
        dVdu_h = self.dVdu_h(u_hat)
        return torch.fft.irfft(dVdu_h)

    def L_hat(self) -> torch.Tensor:
        """
        Linear operator for Implicit SBDF1.
        Must match the linear part dV0du_h.
        """
        return self.a + self.b * (2 * torch.pi) ** 2 * (self.freq) ** 2

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Non-linear part for Explicit SBDF1.
        Must match the nonlinear part dV1du_h.
        """
        return self.dVdu_h(u_hat) - self.L_hat() * u_hat
