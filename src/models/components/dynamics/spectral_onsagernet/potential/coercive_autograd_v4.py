import torch
from torch import nn

from .potential_base import Potential


class CoerciveAutogradPotentialV4(Potential):
    def __init__(
        self,
        Lx: float,
        Nx: int,
        dvdf_h_module: nn.Module,
        positive_a: bool = True,
        positive_b: bool = True,
    ):
        super().__init__()

        self.Lx = Lx
        self.Nx = Nx
        self.dx = Lx / Nx

        self.a_params = nn.Parameter(torch.zeros(1))
        self.b_params = nn.Parameter(torch.zeros(1))

        self.positive_a = positive_a
        self.positive_b = positive_b

        self.dvdf_h_module = dvdf_h_module

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
        return self.a_params**2 if self.positive_a else self.a_params

    @property
    def b(self):
        return self.b_params**2 if self.positive_b else self.b_params

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """
        Baseline Quadratic Potential Energy in Physical Space.
        V0 = ∫ 0.5 * ( a*u^2 + b*(du/dx)^2 ) dx
        """
        u_h = torch.fft.rfft(u, dim=-1)

        grad_u = torch.fft.irfft(1j * self.freq * u_h, n=self.Nx, dim=-1)

        integrand = 0.5 * self.a * u**2 + 0.5 * self.b * grad_u**2

        Vu = torch.sum(integrand, dim=-1) * self.dx  # (B,)
        return Vu.unsqueeze(-1)  # (B, 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """
        Neural Network Correction Potential Energy.
        Calculated in spectral, returned as scalar.
        """
        Vu = u * 0.0  # placeholder
        return Vu.unsqueeze(-1)  # (B, 1)

    def V(self, f: torch.Tensor) -> torch.Tensor:
        """Total Energy for plotting."""
        return self._V0(f) + self._V1(f)

    def _dV0du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Baseline Quadratic Force (Spectral).
        Corresponds to a*I - b*Laplacian
        NO weights here.
        """
        return (self.a + self.b * self.k_sq) * u_hat

    def _dV1du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        u_cat = torch.cat([u_hat.real, u_hat.imag], dim=-1)
        dV1df_cat = self.dvdf_h_module(u_cat)
        dV1du_h = torch.complex(*torch.chunk(dV1df_cat, 2, dim=-1))
        return dV1du_h

    def dVdu_h(self, f_hat: torch.Tensor) -> torch.Tensor:
        """Total Spectral Force."""
        return self._dV0du_h(f_hat) + self._dV1du_h(f_hat)

    def dVdu(self, f: torch.Tensor) -> torch.Tensor:
        """Physical Force (for plotting)."""
        f_hat = torch.fft.rfft(f, dim=-1)
        dVdf_h = self.dVdu_h(f_hat)
        return torch.fft.irfft(dVdf_h, n=self.Nx, dim=-1)

    def regularization_loss(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Regularization loss for the potential parameters."""
        with torch.inference_mode(False):
            # 1. Enable grad tracking for input to compute Jacobian
            u_hat.requires_grad_(True)

            # 2. Forward pass
            dV1du_h = self._dV1du_h(u_hat)  # Output force field

            # 3. Sample random vector v (Gaussian noise)
            v = torch.randn_like(dV1du_h)

            # 4. Compute Vector-Jacobian Product (J^T v)
            # This is standard backprop: v^T * J
            vjp = torch.autograd.grad(
                outputs=dV1du_h,
                inputs=u_hat,
                grad_outputs=v,
                create_graph=True,  # Need graph to optimize weights to fix J
                retain_graph=True,
            )[0]

            # 5. Compute Jacobian-Vector Product (J v)
            # We can use the 'double grad' trick:
            # J v = grad( (mu_out * v).sum(), u_in )? NO.
            # J v = directional derivative in direction v.
            # PyTorch provides a cleaner way now: torch.autograd.functional.jvp
            # But inside a training loop, the "dummy trick" is easiest:

            # TRICK: Jv = grad( (J^T v) * u_dummy ) is complex.
            # EASIER: Use finite difference approximation for Jv (Taylor expansion)
            # Jv \approx (F(u + eps*v) - F(u)) / eps
            # This avoids higher-order derivatives and is very stable.

            epsilon = 1e-3
            dV1du_h_pert = self._dV1du_h(u_hat + epsilon * v)
            jvp_approx = (dV1du_h_pert - dV1du_h) / epsilon

            # 6. Loss: Force Jv == J^T v
            # Scale by N to keep loss magnitude independent of resolution
            sym_loss = torch.mean((jvp_approx - vjp).abs().pow(2))

        return sym_loss

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
