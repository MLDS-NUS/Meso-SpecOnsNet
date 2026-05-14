from abc import ABC, abstractmethod

import torch


class DynamicsBase(ABC, torch.nn.Module):
    def __init__(self, step_method: str = "euler"):
        super().__init__()
        self.step_method = step_method

    @abstractmethod
    def L(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_step(self, u: torch.Tensor, dt: float) -> torch.Tensor:
        if self.step_method == "euler":
            u_next = u + dt * self.L(u)
        elif self.step_method == "rk4":
            k1 = self.L(u)
            k2 = self.L(u + 0.5 * dt * k1)
            k3 = self.L(u + 0.5 * dt * k2)
            k4 = self.L(u + dt * k3)
            u_next = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown step method: {self.step_method}")
        return u_next

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # Define the forward pass
        Lu = self.L(u)

        return Lu
