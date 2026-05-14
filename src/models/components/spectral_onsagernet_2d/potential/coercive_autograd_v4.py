import torch
from torch import nn

from .potential_base import Potential


# 2D port of coercive_autograd_v4.CoerciveAutogradPotentialV4 from the 1D package.
# The residual force dV₁/du_h is produced by an MLP that operates on the
# (real, imag) parts of the full rfft2 spectrum flattened to a 1D feature vector
# of length 2 * Nx * (Ny // 2 + 1).
class CoerciveAutogradPotentialV4(Potential):
    def __init__(
        self,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        dvdf_h_module: nn.Module,
        positive_a: bool = True,
        positive_b: bool = True,
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

        self.positive_a = positive_a
        self.positive_b = positive_b

        self.dvdf_h_module = dvdf_h_module

        # rfft2 layout: x dim is full FFT (fftfreq), y dim is rfft (rfftfreq).
        freq_x = torch.fft.fftfreq(self.Nx, d=self.dx)  # (Nx,)
        freq_y = torch.fft.rfftfreq(self.Ny, d=self.dy)  # (Ny // 2 + 1,)
        kx = 2.0 * torch.pi * freq_x
        ky = 2.0 * torch.pi * freq_y
        self.register_buffer("kx", kx.view(-1, 1))  # (Nx, 1)
        self.register_buffer("ky", ky.view(1, -1))  # (1, Ny // 2 + 1)
        self.register_buffer("k_sq", self.kx**2 + self.ky**2)

        self.kx: torch.Tensor
        self.ky: torch.Tensor
        self.k_sq: torch.Tensor

    @property
    def a(self):
        return self.a_params**2 if self.positive_a else self.a_params

    @property
    def b(self):
        return self.b_params**2 if self.positive_b else self.b_params

    def _V0(self, u: torch.Tensor) -> torch.Tensor:
        """Quadratic baseline V0 = ∫ 0.5 (a u² + b |∇u|²) dx dy."""
        u_h = torch.fft.rfft2(u)
        grad_x = torch.fft.irfft2(1j * self.kx * u_h, s=(self.Nx, self.Ny))
        grad_y = torch.fft.irfft2(1j * self.ky * u_h, s=(self.Nx, self.Ny))

        integrand = 0.5 * self.a * u**2 + 0.5 * self.b * (grad_x**2 + grad_y**2)
        Vu = torch.sum(integrand, dim=(-2, -1)) * self.dx * self.dy  # (..., c)
        return Vu.sum(dim=-1, keepdim=True)  # (..., 1)

    def _V1(self, u: torch.Tensor) -> torch.Tensor:
        """Neural correction V1 has no closed-form energy; placeholder keeps V(u) plottable."""
        return 0.0 * self._V0(u)

    def V(self, u: torch.Tensor) -> torch.Tensor:
        """Total Energy for plotting."""
        return self._V0(u) + self._V1(u)

    def _dV0du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Linear part: a*u + b*(-Δu) in spectral space. No learnable weights here."""
        return (self.a + self.b * self.k_sq) * u_hat

    def _dV1du_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Residual force from a real-valued MLP on the flattened (re, im) spectrum."""
        # u_hat: (..., c, Nx, Ny // 2 + 1) complex
        flat = u_hat.flatten(-2)  # (..., c, Nx * (Ny//2 + 1)) complex
        cat = torch.cat([flat.real, flat.imag], dim=-1)  # (..., c, 2 * Nx * (Ny//2+1))
        out_cat = self.dvdf_h_module(cat)
        out_re, out_im = torch.chunk(out_cat, 2, dim=-1)
        out = torch.complex(out_re.contiguous(), out_im.contiguous())
        return out.unflatten(-1, (self.Nx, self.Ny // 2 + 1))

    def dVdu_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Total Spectral Force."""
        return self._dV0du_h(u_hat) + self._dV1du_h(u_hat)

    def dVdu(self, u: torch.Tensor) -> torch.Tensor:
        """Physical Force (for plotting)."""
        u_hat = torch.fft.rfft2(u)
        dVdu_h = self.dVdu_h(u_hat)
        return torch.fft.irfft2(dVdu_h, s=(self.Nx, self.Ny))

    def regularization_loss(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Symmetry penalty for the residual Jacobian: ‖J v − Jᵀ v‖² (Jv via finite diff)."""
        with torch.inference_mode(False):
            u_hat.requires_grad_(True)

            dV1du_h = self._dV1du_h(u_hat)

            v = torch.randn_like(dV1du_h)

            vjp = torch.autograd.grad(
                outputs=dV1du_h,
                inputs=u_hat,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            epsilon = 1e-3
            dV1du_h_pert = self._dV1du_h(u_hat + epsilon * v)
            jvp_approx = (dV1du_h_pert - dV1du_h) / epsilon

            sym_loss = torch.mean((jvp_approx - vjp).abs().pow(2))

        return sym_loss

    def L_hat(self) -> torch.Tensor:
        """Linear operator for SBDF1; matches _dV0du_h."""
        return self.a + self.b * self.k_sq

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor:
        """Nonlinear operator for SBDF1; matches _dV1du_h."""
        return self._dV1du_h(u_hat)
