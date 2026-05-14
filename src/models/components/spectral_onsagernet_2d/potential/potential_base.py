from abc import ABC, abstractmethod

import torch
from torch import nn


class Potential(nn.Module, ABC):
    @abstractmethod
    def V(self, u: torch.Tensor) -> torch.Tensor:
        """implement V(u): (bL, c, *domain_shape) -> (bL, 1)"""
        pass

    @abstractmethod
    def dVdu(self, u: torch.Tensor) -> torch.Tensor:
        """implement (dV/du)[u]: (bL, c, *domain_shape) -> (bL, c, *domain_shape)"""
        pass

    @abstractmethod
    def dVdu_h(self, u_hat: torch.Tensor) -> torch.Tensor:
        """implement (dV/du)[u_hat]: (bL, c, *freq_shape) -> (bL, c, *freq_shape)"""
        pass

    def L_hat(self) -> torch.Tensor | float:
        return 0.0

    def N_hat(self, u_hat: torch.Tensor) -> torch.Tensor | float:
        return self.dVdu_h(u_hat)
