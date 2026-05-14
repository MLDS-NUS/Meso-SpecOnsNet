"""Classical OnsagerNet treating a 1D PDE grid as a high-dimensional ODE.

Implements the generalized Onsager principle:

    dx/dt = -(M(x) + W(x)) ∇V(x)

where:
    V(x)  — free energy (coercive MLP scalar potential)
    M(x)  — dissipation matrix (symmetric positive semi-definite)
    W(x)  — conservation matrix (anti-symmetric, optional)

Reference: https://github.com/MLDS-NUS/onsagernet-jax
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.dynamics.dynamics_base import DynamicsBase
from src.models.components.utils.activation import Activation

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Simple MLP with configurable hidden dims and activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        act_name: str = "tanh",
        final_bias: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(Activation(act_name))
            prev = h
        layers.append(nn.Linear(prev, out_dim, bias=final_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoercivePotential(nn.Module):
    """Free-energy potential V(x) = (alpha/2)||x||^2 + MLP(x).

    The coercive quadratic term guarantees V → +∞ as ||x|| → ∞,
    matching the standard assumption in the Onsager framework.

    Args:
        in_dim: Dimensionality of the state x.
        hidden_dims: Hidden layer widths of the scalar MLP.
        alpha_init: Initial value of the coercive coefficient α ≥ 0.
        act_name: Activation function name.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        alpha_init: float = 0.01,
        act_name: str = "tanh",
    ):
        super().__init__()
        # final_bias=False: a constant offset in V doesn't affect ∇V (the dynamics),
        # so a final bias would be a permanently unused parameter.
        self.mlp = _MLP(in_dim, hidden_dims, out_dim=1, act_name=act_name, final_bias=False)
        self.log_alpha = nn.Parameter(torch.tensor(alpha_init).log())

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """V(x) — shape (batch,)."""
        quad = 0.5 * self.alpha * (x**2).sum(dim=-1)  # (batch,)
        return quad + self.mlp(x).squeeze(-1)  # (batch,)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """∇V(x) via autograd — shape (batch, in_dim).

        Utility for evaluation / visualization.  Training uses the inlined
        computation in OnsagerNet1d.L() for DDP compatibility.
        """
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            V = self.forward(x_g)
            (g,) = torch.autograd.grad(V.sum(), x_g)
        return g


class DiagonalDissipation(nn.Module):
    """Diagonal dissipation matrix M(x) = diag(β + softplus(m(x))).

    Guarantees SPD by design: diagonal entries are strictly positive.

    Args:
        in_dim: State dimensionality.
        hidden_dims: Hidden widths for the diagonal coefficient MLP.
        beta: Minimum diagonal value (ensures positive-definiteness).
        act_name: Activation for hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        beta: float = 1e-3,
        act_name: str = "tanh",
    ):
        super().__init__()
        self.mlp = _MLP(in_dim, hidden_dims, out_dim=in_dim, act_name=act_name)
        self.beta = beta

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute M(x) @ v efficiently as element-wise product.

        Args:
            x: State, shape (batch, d).
            v: Vector to multiply, shape (batch, d).

        Returns:
            M(x) @ v, shape (batch, d).
        """
        m = self.beta + F.softplus(self.mlp(x))  # (batch, d), positive entries
        return m * v  # element-wise = diag(m) @ v


class LowRankDissipation(nn.Module):
    """Low-rank dissipation matrix M(x) = β·I + B(x)·B(x)ᵀ.

    SPD by construction.  Matrix-vector product costs O(d·r) instead of O(d²).

    Args:
        in_dim: State dimensionality d.
        rank: Rank r of the low-rank factor B.
        hidden_dims: Hidden widths for the factor MLP.
        beta: Minimum eigenvalue (I regularization).
        act_name: Activation for hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        rank: int,
        hidden_dims: Sequence[int],
        beta: float = 1e-3,
        act_name: str = "tanh",
    ):
        super().__init__()
        self.mlp = _MLP(in_dim, hidden_dims, out_dim=in_dim * rank, act_name=act_name)
        self.in_dim = in_dim
        self.rank = rank
        self.beta = beta

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute M(x) @ v = β·v + B(B·v).

        Args:
            x: State, shape (batch, d).
            v: Vector, shape (batch, d).

        Returns:
            M(x) @ v, shape (batch, d).
        """
        B = self.mlp(x).reshape(x.shape[0], self.in_dim, self.rank)  # (batch, d, r)
        Btv = torch.einsum("bdr,bd->br", B, v)  # Bᵀv, (batch, r)
        BBtv = torch.einsum("bdr,br->bd", B, Btv)  # B(Bᵀv), (batch, d)
        return self.beta * v + BBtv


class LowRankConservation(nn.Module):
    """Low-rank anti-symmetric conservation matrix W(x) = B(x)·Cᵀ − C·B(x)ᵀ.

    Anti-symmetric by construction (W = −Wᵀ).

    Args:
        in_dim: State dimensionality d.
        rank: Rank r of the low-rank factors.
        hidden_dims: Hidden widths for the B-factor MLP.
        act_name: Activation for hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        rank: int,
        hidden_dims: Sequence[int],
        act_name: str = "tanh",
    ):
        super().__init__()
        self.mlp = _MLP(in_dim, hidden_dims, out_dim=in_dim * rank, act_name=act_name)
        # Learnable fixed factor C ∈ ℝ^{d×r}
        self.C = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.in_dim = in_dim
        self.rank = rank

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute W(x) @ v = B(Cᵀv) − C(Bᵀv).

        Args:
            x: State, shape (batch, d).
            v: Vector, shape (batch, d).

        Returns:
            W(x) @ v, shape (batch, d).
        """
        B = self.mlp(x).reshape(x.shape[0], self.in_dim, self.rank)  # (batch, d, r)
        Ctv = torch.einsum("dr,bd->br", self.C, v)  # Cᵀv, (batch, r)
        Btv = torch.einsum("bdr,bd->br", B, v)  # Bᵀv, (batch, r)
        BCt_v = torch.einsum("bdr,br->bd", B, Ctv)  # B(Cᵀv), (batch, d)
        CBt_v = torch.einsum("dr,br->bd", self.C, Btv)  # C(Bᵀv), (batch, d)
        return BCt_v - CBt_v


# ---------------------------------------------------------------------------
# Circulant (convolution-structured) variants
# ---------------------------------------------------------------------------


class CirculantDissipation(nn.Module):
    """State-independent circulant SPD dissipation matrix.

    Implements M via a learned symmetric convolution kernel:

        M·v = IRFFT(softplus(θ) ⊙ RFFT(v)) + β·v

    A circulant matrix is fully described by its eigenvalues in the Fourier
    basis.  For a symmetric positive-definite circulant, those eigenvalues must
    be real and positive — enforced here by parameterizing them through
    ``softplus``.  The ``band_size`` argument biases initialization toward a
    spatially-localized (short-range) kernel; without it the kernel is
    initialized to be uniform across all frequencies.

    Unlike ``DiagonalDissipation`` and ``LowRankDissipation``, the matrix does
    **not** depend on the state *x* — it is a fixed learned linear operator,
    matching the structure assumed in ``SpectralOnsagerNet1d``.

    Args:
        Nx: Number of grid points (= state dimension for single-channel input).
        beta: Minimum eigenvalue added after softplus (ensures strict SPD).
        band_size: If given, initialize the spatial kernel as a Gaussian of
            width ``band_size`` grid points, so the effective range starts
            local.  Must satisfy ``1 <= band_size <= Nx``.
    """

    def __init__(self, Nx: int, beta: float = 1e-3, band_size: int | None = None):
        super().__init__()
        self.Nx = Nx
        self.beta = beta
        freq_len = Nx // 2 + 1

        if band_size is None:
            init = torch.zeros(freq_len)  # softplus(0) ≈ 0.693 for every mode
        else:
            # Build a wrapped Gaussian kernel of width ~ band_size in physical space.
            # Wrap index: circular distance = min(n, Nx - n).
            idx = torch.arange(Nx, dtype=torch.float32)
            dist = torch.minimum(idx, Nx - idx)  # circular distance
            sigma = max(band_size / 4.0, 1.0)
            kernel = torch.exp(-0.5 * (dist / sigma) ** 2)
            kernel = kernel / kernel.sum()  # normalize to unit mass
            # Eigenvalues = RFFT of the (symmetric) kernel → real-valued.
            lam = torch.fft.rfft(kernel).real.clamp(min=1e-6)
            # Invert softplus: θ = log(exp(λ) − 1)  [i.e. softplus(θ) = λ]
            init = torch.log(torch.expm1(lam))

        self.theta = nn.Parameter(init)

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute M·v = IRFFT(softplus(θ) ⊙ RFFT(v)) + β·v.

        Args:
            x: State, shape (batch, d).  Unused — M is state-independent.
            v: Vector, shape (batch, d).

        Returns:
            M·v, shape (batch, d).
        """
        v_hat = torch.fft.rfft(v, dim=-1)  # (batch, freq_len) complex
        eigenvalues = F.softplus(self.theta)  # (freq_len,) real positive
        Mv_hat = eigenvalues * v_hat
        return torch.fft.irfft(Mv_hat, n=v.shape[-1], dim=-1) + self.beta * v


class CirculantConservation(nn.Module):
    """State-independent circulant anti-symmetric conservation matrix.

    Implements W via a learned antisymmetric convolution kernel:

        W·v = IRFFT(i · θ ⊙ RFFT(v))

    A real antisymmetric circulant has purely imaginary Fourier eigenvalues.
    Parameterizing them as ``i · θ`` (θ real) enforces antisymmetry by
    construction: W = −Wᵀ.  The DC (k=0) and Nyquist (k=Nx//2) components are
    held at zero because a real antisymmetric kernel must have zero mean and
    zero contribution at the Nyquist frequency.

    Like ``CirculantDissipation``, this matrix is state-independent.

    Args:
        Nx: Number of grid points.
        band_size: If given, initialize θ to be non-zero only for frequencies
            ``k ∈ [1, band_size // 2]``, so the initial kernel is short-range.
            Without it, θ is zero-initialized (no conservation at startup).
    """

    def __init__(self, Nx: int, band_size: int | None = None):
        super().__init__()
        self.Nx = Nx
        freq_len = Nx // 2 + 1

        init = torch.zeros(freq_len)
        if band_size is not None:
            cutoff = max(1, band_size // 2)
            cutoff = min(cutoff, freq_len - 2)  # never touch DC or Nyquist
            init[1 : cutoff + 1] = 0.01

        self.theta = nn.Parameter(init)

        # Mask to zero out DC (k=0) and Nyquist (k=Nx//2) permanently.
        mask = torch.ones(freq_len)
        mask[0] = 0.0
        mask[-1] = 0.0
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute W·v = IRFFT(i · (mask ⊙ θ) ⊙ RFFT(v)).

        Args:
            x: State, shape (batch, d).  Unused — W is state-independent.
            v: Vector, shape (batch, d).

        Returns:
            W·v, shape (batch, d).
        """
        v_hat = torch.fft.rfft(v, dim=-1)  # (batch, freq_len) complex
        theta_masked = self.theta * self.mask  # zero DC and Nyquist
        Wv_hat = 1j * theta_masked * v_hat
        return torch.fft.irfft(Wv_hat, n=v.shape[-1], dim=-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class OnsagerNet1d(DynamicsBase):
    """Classical OnsagerNet for 1D PDE grids treated as high-dimensional ODEs.

    The PDE solution u(x, t) sampled on an Nx-point grid is treated as a
    vector in ℝ^(c·Nx), and the classical Onsager dynamics are applied:

        du/dt = -(M(u) + W(u)) ∇V(u)

    Compatible with the MesoLitModule framework: accepts tensors of shape
    (batch, c, Nx) and returns the same shape.

    Args:
        Nx: Number of spatial grid points.
        n_channels: Number of field channels (default 1).
        potential: CoercivePotential module.
        dissipation: DiagonalDissipation or LowRankDissipation module.
        conservation: Optional LowRankConservation module.
        step_method: Time integration method ("euler" or "rk4").
    """

    def __init__(
        self,
        Nx: int,
        n_channels: int,
        potential: CoercivePotential,
        dissipation: DiagonalDissipation | LowRankDissipation,
        conservation: LowRankConservation | None = None,
        step_method: str = "euler",
    ):
        super().__init__(step_method=step_method)
        self.Nx = Nx
        self.n_channels = n_channels
        self.potential = potential
        self.dissipation = dissipation
        self.conservation = conservation

    def L(self, u: torch.Tensor) -> torch.Tensor:
        """Compute RHS -(M(x) + W(x)) ∇V(x).

        Args:
            u: Field tensor of shape (batch, c, Nx).

        Returns:
            Time derivative of shape (batch, c, Nx).
        """
        b, c, Nx = u.shape
        x = u.reshape(b, c * Nx)  # (batch, d)

        # Compute ∇V via autograd.  torch.enable_grad() ensures autograd is
        # active even when called inside torch.no_grad() (e.g. Lightning's
        # validation_step) or torch.compile contexts.  Calling self.potential()
        # here (not inside a helper) keeps DDP parameter tracking working.
        with torch.inference_mode(False):
            x_g = x.detach().requires_grad_(True)
            V = self.potential(x_g)  # (batch,)
            (grad_V,) = torch.autograd.grad(
                V.sum(),
                x_g,
                create_graph=self.training,
                retain_graph=self.training,
            )  # (batch, d)

        rhs = -self.dissipation.apply(x, grad_V)
        if self.conservation is not None:
            rhs = rhs - self.conservation.apply(x, grad_V)

        return rhs.reshape(b, c, Nx)
