import torch
from torch import nn

from .potential import Potential


class SpectralOnsagerNet2d(nn.Module):
    def __init__(
        self,
        learn_M: bool,
        learn_W: bool,
        potential: Potential,
        Lx: float = 2.0 * torch.pi,
        Ly: float = 2.0 * torch.pi,
        Nx: int = 128,
        Ny: int = 128,
        method: str = "euler",
    ):
        super().__init__()

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        self.M_coeff = 1.0 if learn_M else 0.0
        self.W_coeff = 1.0 if learn_W else 0.0

        self.method = method

        # rfft2 output shape on (..., Nx, Ny): (..., Nx, Ny // 2 + 1)
        freq_len_x = self.Nx
        freq_len_y = self.Ny // 2 + 1

        # 2D frequency space parameters
        self.Km_params = nn.Parameter(torch.randn(freq_len_x, freq_len_y) * 0.1)
        self.Kw_params = nn.Parameter(torch.randn(freq_len_x, freq_len_y) * 0.01)

        self.potential = potential

    @property
    def Km(self):
        return self.M_coeff * self.Km_params.square()

    @property
    def Kw(self):
        return self.W_coeff * self.Kw_params * 1j

    def rhs_spectral(self, u_hat: torch.Tensor) -> torch.Tensor:
        dVdu_h = self.potential.dVdu_h(u_hat)  # (bs, c, Nx, Ny//2+1)
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

        Args:
            u_hat: Fourier coefficients of the solution
            dt: Time step size

        Returns:
            Updated Fourier coefficients after one time step
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
