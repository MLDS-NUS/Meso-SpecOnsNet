import torch
from torch import nn

from .potential import Potential


class SpectralOnsagerNetAllenCahn2d(nn.Module):
    """2D Allen-Cahn dynamics with fixed mobility Km=1 and Kw=0.

    Physics-informed counterpart of SpectralOnsagerNet2d: the Onsager operator
    is fixed to the identity (no learnable M, W) so that pairing it with the
    analytical AllenCahnPotential reproduces u_t = Δu + (u - u³)/ε² exactly.
    """

    def __init__(
        self,
        potential: Potential,
        Lx: float = 2.0 * torch.pi,
        Ly: float = 2.0 * torch.pi,
        Nx: int = 128,
        Ny: int = 128,
        method: str = "sbdf1",
    ):
        super().__init__()

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        self.M_coeff = 1.0
        self.W_coeff = 1.0

        self.method = method

        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.potential = potential

    @property
    def Km(self):
        return 1.0 + 0.0 * self.dummy_param

    @property
    def Kw(self):
        return 0.0

    def rhs_spectral(self, u_hat: torch.Tensor) -> torch.Tensor:
        dVdu_h = self.potential.dVdu_h(u_hat)
        return -(self.Km + self.Kw) * dVdu_h

    def _sbdf1_stepper(self, u_hat: torch.Tensor, dt: float) -> torch.Tensor:
        """Semi-implicit backward differentiation formula of order 1.

        Implements the SBDF1 time-stepping scheme:
            - u⁺ - u = -(M + W)[μ]Δt
            - û⁺ - û = -(Km + Kw)μ̂Δt

        Decomposition: μ̂ = L̂ * û + N̂(û) [linear + nonlinear]

        Solving for û⁺:
            - û⁺ - û = -(Km + Kw)(L̂ * û⁺ + N̂(û))Δt
            - û⁺ = [û - Δt(Km + Kw)N̂(û)] / [1 + Δt(Km + Kw)L̂]
        """
        Nh_uh = self.potential.N_hat(u_hat)
        Lh = self.potential.L_hat()
        return (u_hat - dt * (self.Km + self.Kw) * Nh_uh) / (1.0 + dt * (self.Km + self.Kw) * Lh)

    def forward_step_spectral(self, u_hat: torch.Tensor, dt: float) -> torch.Tensor:
        if self.method == "euler":
            k1 = self.rhs_spectral(u_hat)
            return u_hat + dt * k1
        elif self.method == "rk2":
            k1 = self.rhs_spectral(u_hat)
            k2 = self.rhs_spectral(u_hat + dt * k1)
            return u_hat + dt * 0.5 * (k1 + k2)
        elif self.method == "rk3":
            # ref: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
            k1 = self.rhs_spectral(u_hat)
            k2 = self.rhs_spectral(u_hat + dt * 0.5 * k1)
            k3 = self.rhs_spectral(u_hat + dt * (-k1 + 2 * k2))
            return u_hat + dt * (k1 + 4 * k2 + k3) / 6
        elif self.method == "sbdf1":
            return self._sbdf1_stepper(u_hat, dt)
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

    def forward_step(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        # u.shape: (..., Nx, Ny)
        u_hat = torch.fft.rfft2(u)
        u_hat = self.forward_step_spectral(u_hat, dt)
        u = torch.fft.irfft2(u_hat, s=(self.Nx, self.Ny))
        return u

    def regularization_loss(self, u: torch.Tensor) -> torch.Tensor:
        if hasattr(self.potential, "regularization_loss"):
            u_hat = torch.fft.rfft2(u)
            return self.potential.regularization_loss(u_hat)
        return torch.zeros(u.shape[0], device=u.device)
