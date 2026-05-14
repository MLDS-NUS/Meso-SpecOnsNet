import torch


def ddx(u: torch.Tensor, dx: float, dim: int = -1, method: str = "c") -> torch.Tensor:
    up = torch.roll(u, -1, dims=dim)
    un = torch.roll(u, 1, dims=dim)
    upp = torch.roll(u, -2, dims=dim)
    unn = torch.roll(u, 2, dims=dim)

    if method == "c":
        dfdx = (up - un) / (2 * dx)
    elif method == "p":
        dfdx = (up - u) / dx
    elif method == "n":
        dfdx = (u - un) / dx
    elif method == "p2":
        dfdx = (-3 * u + 4 * up - upp) / (2 * dx)
    elif method == "n2":
        dfdx = (3 * u - 4 * un + unn) / (2 * dx)
    else:
        raise ValueError(f"Unknown method: {method}")

    return dfdx  # (..., Nx, Nv)


def ddx_upwind(u: torch.Tensor, dx: float, indicator: torch.Tensor | float, dim: int = -1) -> torch.Tensor:
    u_p = torch.roll(u, shifts=-1, dims=dim)
    u_m = torch.roll(u, shifts=1, dims=dim)

    if isinstance(indicator, float):
        indicator = torch.full_like(u, indicator)
    dudx = torch.where(indicator >= 0, (u - u_m) / dx, (u_p - u) / dx)
    return dudx


def d2dx2(u: torch.Tensor, dx: float, dim: int = -1) -> torch.Tensor:
    u_p = torch.roll(u, shifts=-1, dims=dim)
    u_m = torch.roll(u, shifts=1, dims=dim)

    return (u_p - 2 * u + u_m) / (dx**2)
