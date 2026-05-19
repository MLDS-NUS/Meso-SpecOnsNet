"""Shared helpers for the component-scaling notebooks (kdv / fpu / fene).

Each notebook only specifies its dataset key, display name, trajectory indices,
fit window, and Fmap y-limits — loading, V-component sweep, slope fitting, and
plotting are done here.
"""

from __future__ import annotations

import functools
import glob
import re
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from src.models.meso_module import MesoLitModule

DATASET_CONFIG: dict[str, dict] = {
    "kdv": {"data_subdir": "kdv", "display_name": "KdV"},
    "ckdv": {"data_subdir": "fput", "display_name": "FPUT"},
    "fene_v2": {"data_subdir": "fene", "display_name": "FENE"},
}


def find_latest_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts = sorted(
            [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"],
            key=lambda p: p.stat().st_mtime,
        )
        if ckpts:
            return ckpts[-1]
    ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return ckpts[-1]


def load_model_for_inference(run_dir: Path, root: Path, device: str = "cpu") -> MesoLitModule:
    config_file = run_dir / ".hydra" / "config.yaml"
    cfg_text = config_file.read_text()
    # Runs in logs/official/ predate the refactor that moved the spectral
    # OnsagerNet 1d package under dynamics/; rewrite the legacy _target_ path.
    cfg_text = cfg_text.replace(
        "src.models.components.spectral_onsagernet.",
        "src.models.components.dynamics.spectral_onsagernet.",
    )
    cfg_raw = yaml.safe_load(cfg_text)

    dynamics_cfg = OmegaConf.create(cfg_raw["model"]["dynamics"])
    dynamics = hydra.utils.instantiate(dynamics_cfg)

    model_cfg = cfg_raw["model"]
    model = MesoLitModule(
        dynamics=dynamics,
        optimizer=functools.partial(torch.optim.Adam),
        dt=model_cfg["dt"],
        compile=False,
        accumulated_nsteps=model_cfg.get("accumulated_nsteps", 1),
        loss_fn=model_cfg.get("loss_fn", "mse"),
        reg_weight=model_cfg.get("reg_weight"),
    )

    ckpt_path = find_latest_ckpt(run_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {re.sub(r"\._orig_mod\.", ".", k): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    print(f"  Loaded: {ckpt_path.relative_to(root)}")
    return model


def load_test_data(data_glob: str, split_ratio: tuple = (0.7, 0.2, 0.1)) -> tuple:
    files = sorted(glob.glob(data_glob))
    trajs, t_coord, x_coord = [], None, None
    for fp in files:
        with h5py.File(fp, "r") as f:
            u = f["u_sol_all"][:]
            if t_coord is None and "t_coord" in f:
                t_coord = f["t_coord"][:]
            if x_coord is None and "x_coord" in f:
                x_coord = f["x_coord"][:]
        trajs.append(u.reshape(-1, *u.shape[-3:]))
    u_all = np.concatenate(trajs, axis=0)
    N = len(u_all)
    val_end = int(N * (split_ratio[0] + split_ratio[1]))
    u_test = np.transpose(u_all[val_end:], (0, 1, 3, 2))  # (N_test, T, n_vars, Nx)
    return torch.from_numpy(u_test).float(), t_coord, x_coord


def _components(potential, u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-term potential for a batch u: (B, n_vars, Nx) → dict of (B,) tensors."""
    B = u.shape[0]
    V0 = potential._V0(u).reshape(B)
    V1 = potential._V1(u).reshape(B)
    V2 = potential._V2(u).reshape(B)
    return {"V0": V0, "V1": V1, "V2": V2, "V": V0 + V1 + V2}


def run_scaling_sweep(
    potential,
    u0_batch: torch.Tensor,
    a_values: np.ndarray,
    n_vars: int,
    Nx: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Sweep V_theta(a · u_0) for each (a, traj) pair, shifted by V_theta(0).

    Returns a dict with arrays of shape (N_A, N_traj) under keys
    'V', 'V0', 'V1', 'V2' (each is the dV-from-zero version).
    """
    N_A = len(a_values)
    N_traj = u0_batch.shape[0]

    with torch.no_grad():
        zero_batch = torch.zeros((1, n_vars, Nx), device=device)
        V_at_zero = {k: v.item() for k, v in _components(potential, zero_batch).items()}
    print("V(0):", {k: f"{v:.4e}" for k, v in V_at_zero.items()})

    V_vals = np.zeros((N_A, N_traj), dtype=np.float64)
    V0_vals = np.zeros((N_A, N_traj), dtype=np.float64)
    V1_vals = np.zeros((N_A, N_traj), dtype=np.float64)
    V2_vals = np.zeros((N_A, N_traj), dtype=np.float64)

    with torch.no_grad():
        for i, a in enumerate(a_values):
            comps = _components(potential, float(a) * u0_batch)
            V_vals[i] = comps["V"].cpu().numpy()
            V0_vals[i] = comps["V0"].cpu().numpy()
            V1_vals[i] = comps["V1"].cpu().numpy()
            V2_vals[i] = comps["V2"].cpu().numpy()

    return {
        "V": V_vals - V_at_zero["V"],
        "V0": V0_vals - V_at_zero["V0"],
        "V1": V1_vals - V_at_zero["V1"],
        "V2": V2_vals - V_at_zero["V2"],
    }


def fit_slopes(a_values: np.ndarray, dV: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trajectory log-log slope of |dV| against a (within fit_mask).

    Returns (slopes, intercepts, r2), each of shape (N_traj,).
    """
    N_traj = dV.shape[1]
    slopes = np.full(N_traj, np.nan)
    intercepts = np.full(N_traj, np.nan)
    r2 = np.full(N_traj, np.nan)
    for k in range(N_traj):
        y = np.abs(dV[fit_mask, k])
        mask = y > 0
        if mask.sum() < 2:
            continue
        lx = np.log(a_values[fit_mask][mask])
        ly = np.log(y[mask])
        slope, intercept = np.polyfit(lx, ly, 1)
        slopes[k] = slope
        intercepts[k] = intercept
        y_pred = slope * lx + intercept
        ss_res = np.sum((ly - y_pred) ** 2)
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        r2[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slopes, intercepts, r2


def plot_total_V_scaling(
    a_values: np.ndarray,
    dV: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    traj_idxs: list[int],
    display_name: str,
    a_min: float,
    a_max: float,
    a_fit_max: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")

    for k, ti in enumerate(traj_idxs):
        color = cmap(k % 10)
        y = np.abs(dV[:, k])
        pos = y > 0
        ax.plot(a_values[pos], y[pos], "o", color=color, markersize=3.5, label=rf"traj {ti}  ($p$={slopes[k]:.2f})")
        if np.isfinite(slopes[k]):
            ys = np.exp(intercepts[k]) * a_values ** slopes[k]
            ax.plot(a_values, ys, "-", color=color, lw=1.0, alpha=0.6)

    ax.axvline(1.0, color="k", lw=0.6, linestyle=":")
    if a_fit_max < a_max:
        ax.axvline(a_fit_max, color="gray", lw=0.8, linestyle="--", label=rf"fit window $a<{a_fit_max}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scalar $a$", fontsize=12)
    ax.set_ylabel(r"$|V_\theta(a\,u_0) - V_\theta(0)|$", fontsize=12)
    ax.set_title(rf"{display_name} — potential order via input scaling", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_components_scaling(
    a_values: np.ndarray,
    dV_by_name: dict[str, np.ndarray],
    slopes_by_name: dict[str, np.ndarray],
    intercepts_by_name: dict[str, np.ndarray],
    r2_by_name: dict[str, np.ndarray],
    traj_idxs: list[int],
    display_name: str,
    a_min: float,
    a_max: float,
    a_fit_max: float,
    out_path: Path,
    skip_fit_line: set[str] = frozenset({"V_2"}),
) -> None:
    """Plot one panel per component name in dV_by_name (markers + optional fit lines)."""
    palette = sns.color_palette("deep")
    names = list(dV_by_name.keys())
    n_panels = len(names)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        dV = dV_by_name[name]
        slopes = slopes_by_name[name]
        inters = intercepts_by_name[name]
        r2 = r2_by_name[name]
        for k, _ti in enumerate(traj_idxs):
            color = palette[k % len(palette)]
            y = np.abs(dV[:, k])
            pos = y > 0
            if name in skip_fit_line:
                label = rf"$u_0$ #{k + 1}"
            else:
                label = rf"$u_0$ #{k + 1}  ($\alpha$={slopes[k]:.2f}, $R^2$={r2[k]:.3f})"
            ax.plot(a_values[pos], y[pos], "o", color=color, markersize=3.0, label=label)
            if np.isfinite(slopes[k]) and name not in skip_fit_line:
                ys = np.exp(inters[k]) * a_values ** slopes[k]
                ax.plot(a_values, ys, "-", color=color, lw=1.0, alpha=0.6)

        ax.axvline(1.0, color="k", lw=0.6, linestyle=":")
        if a_fit_max < a_max and name not in skip_fit_line:
            ax.axvline(a_fit_max, color="gray", lw=0.8, linestyle="--", label=rf"fit window: $a<{a_fit_max}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(a_min, a_max)
        ax.set_ylim(bottom=1e-2)
        ax.set_xlabel(r"scalar $a$", fontsize=16)
        ax.set_ylabel(rf"$|{name}(a\,u_0) - {name}(0)|$", fontsize=16)
        ax.set_title(rf"{display_name} — ${name}$ scaling", fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=12, loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_V0_scaling(
    a_values: np.ndarray,
    dV0: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    r2: np.ndarray,
    traj_idxs: list[int],
    display_name: str,
    a_min: float,
    a_max: float,
    a_fit_max: float,
    out_path: Path,
) -> None:
    """Plot only the V_0 component scaling (single panel, markers + fit lines)."""
    palette = sns.color_palette("deep")
    fig, ax = plt.subplots(figsize=(6, 4))

    for k, _ti in enumerate(traj_idxs):
        color = palette[k % len(palette)]
        y = np.abs(dV0[:, k])
        pos = y > 0
        label = rf"$u_0$ #{k + 1}  ($\alpha$={slopes[k]:.2f}, $R^2$={r2[k]:.3f})"
        ax.plot(a_values[pos], y[pos], "o", color=color, markersize=3.0, label=label)
        if np.isfinite(slopes[k]):
            ys = np.exp(intercepts[k]) * a_values ** slopes[k]
            ax.plot(a_values, ys, "-", color=color, lw=1.0, alpha=0.6)

    ax.axvline(1.0, color="k", lw=0.6, linestyle=":")
    if a_fit_max < a_max:
        ax.axvline(a_fit_max, color="gray", lw=0.8, linestyle="--", label=rf"fit window: $a<{a_fit_max}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel(r"scalar $a$", fontsize=14)
    ax.set_ylabel(r"$|V_0(a\,u_0) - V_0(0)|$", fontsize=14)
    ax.set_title(rf"{display_name} — $V_0$ scaling", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_Fmap(
    potential,
    test_data: torch.Tensor,
    n_vars: int,
    display_name: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
    box_ylim: tuple[float, float] | None = None,
    u_max_factor: float = 3.0,
    n_grid: int = 401,
    device: str = "cpu",
) -> None:
    """Plot the pointwise nonlinear energy density (F(u) - F(0))/a."""
    assert n_vars == 1, "F(u) plot assumes a single-channel field (n_vars == 1)."

    u_data = test_data[:, :, 0, :].numpy().ravel()
    u_data_max = float(np.abs(u_data).max())
    U_MAX = u_max_factor * u_data_max
    u_grid = np.linspace(-U_MAX, U_MAX, n_grid)
    print(f"u grid: {n_grid} points in [{-U_MAX:.3f}, {U_MAX:.3f}]  (data range: |u|_max = {u_data_max:.3f})")

    u_in = torch.from_numpy(u_grid).float().reshape(n_grid, 1, 1).to(device)
    u_zero = torch.zeros((1, 1, 1), device=device)
    with torch.no_grad():
        F_raw = potential.Fmap(u_in).cpu().numpy().reshape(n_grid)
        F0 = float(potential.Fmap(u_zero).cpu().item())
        aF = float(potential.a.detach().cpu().item())

    F_vals = F_raw - F0  # / aF
    print(f"F(0) = {F0:.4e}  →  subtracted so the curve passes through the origin")
    print(f"Quadratic coefficient a (softplus): {aF:.4e}  →  divided out of the curve")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(u_grid, F_vals, "-", lw=1.6, label=r"$F(u) - F(0)$")
    # ax.plot(u_grid, 0.5 * u_grid**2, "--", lw=1.0, color="gray", label=r"$\frac{1}{2} u^2$")
    ax.plot(
        u_grid,
        aF * 0.5 * u_grid**2 + F_vals,
        ":",
        lw=1.2,
        color="C3",
        label=r"$V_0(u) - V_0(0)$",
    )
    for xv in (-u_data_max, u_data_max):
        ax.axvline(xv, color="k", lw=0.5, linestyle=":", alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlim(-1, 1)

    if box_ylim is not None:
        box_x0, box_x1 = -0.3, 0.3
        box_y0, box_y1 = box_ylim
        ax.add_patch(
            Rectangle(
                (box_x0, box_y0),
                box_x1 - box_x0,
                box_y1 - box_y0,
                fill=False,
                edgecolor="red",
                lw=1.2,
                linestyle="--",
                label=r"region where $u$ is small",
            )
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$u$", fontsize=14)
    ax.set_ylabel(r"energy density (shifted)", fontsize=14)
    ax.set_title(rf"{display_name} — pointwise density of $V_0$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
