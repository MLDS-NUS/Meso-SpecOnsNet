"""Shared helpers for the one-step-potential-variation Δt-scaling notebooks.

Each notebook only needs to specify its dataset key and trajectory indices —
the rollout sweep and plotting are done here.
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
from matplotlib.ticker import FixedLocator, FuncFormatter
from omegaconf import OmegaConf

from src.models.meso_module import MesoLitModule

DATASET_CONFIG = {
    "kdv": {"data_subdir": "kdv", "train_dt": 1e-3, "T_final": 0.1, "display": "KdV", "n_spatial_dims": 1},
    "ckdv": {"data_subdir": "fput", "train_dt": 1e-2, "T_final": 1.0, "display": "FPU chain", "n_spatial_dims": 1},
    "fene_v2": {
        "data_subdir": "fene",
        "train_dt": 5e-3,
        "T_final": 0.5,
        "display": "FENE chain",
        "n_spatial_dims": 1,
    },
    "ac": {"data_subdir": "allen_cahn", "train_dt": 1e-3, "T_final": 0.1, "display": "Allen-Cahn", "n_spatial_dims": 1},
    "ac_2d": {
        "data_subdir": "allen_cahn_2d",
        "train_dt": 1e-3,
        "T_final": 0.1,
        "display": "Allen-Cahn 2D",
        "n_spatial_dims": 2,
        "data_glob": "*1000_seed0.hdf5",
    },
}


def find_latest_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        # Exclude Lightning's "last*.ckpt" family (last.ckpt, last-v1.ckpt, ...) —
        # these are auto-versioned save_last snapshots from resumed runs and are
        # not tied to a specific epoch.
        ckpts = sorted(
            [p for p in ckpt_dir.glob("*.ckpt") if not p.name.startswith("last")],
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


def load_test_data_nd(
    data_glob: str,
    n_spatial_dims: int = 1,
    split_ratio: tuple = (0.7, 0.2, 0.1),
) -> tuple:
    """N-D variant of load_test_data — output (N_test, T, n_vars, *spatial)."""
    files = sorted(glob.glob(data_glob))
    trajs, t_coord, x_coord, y_coord = [], None, None, None
    keep = 1 + n_spatial_dims + 1  # T + spatial + n_vars
    for fp in files:
        with h5py.File(fp, "r") as f:
            u = f["u_sol_all"][:]
            if t_coord is None and "t_coord" in f:
                t_coord = f["t_coord"][:]
            if x_coord is None and "x_coord" in f:
                x_coord = f["x_coord"][:]
            if y_coord is None and "y_coord" in f:
                y_coord = f["y_coord"][:]
        trajs.append(u.reshape(-1, *u.shape[-keep:]))

    u_all = np.concatenate(trajs, axis=0)  # (N, T, *spatial, n_vars)
    N = len(u_all)
    val_end = int(N * (split_ratio[0] + split_ratio[1]))
    u_test = u_all[val_end:]
    u_test = np.moveaxis(u_test, -1, 2)  # → (N_test, T, n_vars, *spatial)
    return torch.from_numpy(u_test).float(), t_coord, x_coord, y_coord


def _compute_V(potential, u: torch.Tensor) -> torch.Tensor:
    """Total potential V(u) for a batch u: (B, n_vars, Nx) → (B,).

    `potential.V` has a broadcasting bug under batching ((B,1)+(B,1,1) → (B,B,1));
    we bypass it by reshaping each component to (B,) before summing.
    """
    B = u.shape[0]
    V0 = potential._V0(u).reshape(B)
    V1 = potential._V1(u).reshape(B)
    V2 = potential._V2(u).reshape(B)
    return V0 + V1 + V2


def make_dt_plan(train_dt: float, T_final: float, n_dt: int = 5) -> list[tuple[float, int]]:
    """Log-spaced Δt values from train_dt/10 up to train_dt."""
    dt_values = np.logspace(np.log10(train_dt * 0.1), np.log10(train_dt), n_dt)
    return [(float(dt), max(2, int(round(T_final / dt)))) for dt in dt_values]


def run_dt_sweep(
    model: MesoLitModule,
    test_data: torch.Tensor,
    traj_idxs: list[int],
    plan: list[tuple[float, int]],
    device: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the model out at each Δt and record max/median |V_{k+1} - V_k|.

    Returns
    -------
    max_step_dV    : (N_dt, N_traj)
    median_step_dV : (N_dt, N_traj)
    finite         : (N_dt, N_traj) bool — True if rollout stayed finite
    """
    potential = model.dynamics.potential
    u0_batch = test_data[traj_idxs, 0].to(device)
    N_traj = u0_batch.shape[0]

    max_step_dV = np.full((len(plan), N_traj), np.nan, dtype=np.float64)
    median_step_dV = np.full((len(plan), N_traj), np.nan, dtype=np.float64)
    finite = np.zeros((len(plan), N_traj), dtype=bool)

    with torch.no_grad():
        for i, (dt, n_steps) in enumerate(plan):
            x = u0_batch.clone()
            V_prev = _compute_V(potential, x)
            step_dVs = torch.zeros(n_steps, N_traj, device=device)
            bad = torch.zeros(N_traj, dtype=torch.bool, device=device)

            for s in range(n_steps):
                x = model.dynamics.forward_step(x, dt=dt)
                new_bad = ~torch.isfinite(x).all(dim=(-2, -1))
                bad = bad | new_bad
                if new_bad.any():
                    x = torch.where(new_bad[:, None, None], u0_batch, x)
                V = _compute_V(potential, x)
                step_dVs[s] = (V - V_prev).abs()
                V_prev = V

            good = ~bad
            per_traj_max = step_dVs.max(dim=0).values
            per_traj_median = step_dVs.median(dim=0).values
            nan_t = torch.full_like(per_traj_max, float("nan"))
            max_step_dV[i] = torch.where(good, per_traj_max, nan_t).cpu().numpy()
            median_step_dV[i] = torch.where(good, per_traj_median, nan_t).cpu().numpy()
            finite[i] = good.cpu().numpy()

            if verbose:
                tagged = " ".join(
                    f"{traj_idxs[k]}:{max_step_dV[i, k]:.2e}/{median_step_dV[i, k]:.2e}" + ("" if finite[i, k] else "!")
                    for k in range(N_traj)
                )
                print(f"Δt={dt:.2e}  n_steps={n_steps:>5d}  max/med = {tagged}")

    return max_step_dV, median_step_dV, finite


def _latex_sci(x: float) -> str:
    """Format x as a LaTeX scientific-notation string, e.g. 5e-3 → '5\\times10^{-3}'."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"
    if np.isclose(mant, round(mant)):
        return rf"{int(round(mant))}\times10^{{{exp}}}"
    return rf"{mant:.2g}\times10^{{{exp}}}"


def _fit_powerlaw(
    dt_arr: np.ndarray, data: np.ndarray, finite: np.ndarray, fit_cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Per-trajectory log-log fit of |ΔV| ~ C · Δt^α, restricted to |ΔV| < fit_cap."""
    N_traj = data.shape[1]
    slopes = np.full(N_traj, np.nan)
    intercepts = np.full(N_traj, np.nan)
    r2s = np.full(N_traj, np.nan)
    fit_masks: list[np.ndarray] = []
    excluded_masks: list[np.ndarray] = []

    for k in range(N_traj):
        valid_mask = finite[:, k] & (data[:, k] > 0)
        fit_mask = valid_mask & (data[:, k] < fit_cap)
        fit_masks.append(fit_mask)
        excluded_masks.append(valid_mask & ~fit_mask)
        if fit_mask.sum() < 2:
            continue
        x = np.log(dt_arr[fit_mask])
        y = np.log(data[fit_mask, k])
        slope, intercept = np.polyfit(x, y, 1)
        slopes[k] = slope
        intercepts[k] = intercept
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2s[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slopes, intercepts, r2s, fit_masks, excluded_masks


def plot_max_and_median(
    dt_arr: np.ndarray,
    max_step_dV: np.ndarray,
    median_step_dV: np.ndarray,
    finite: np.ndarray,
    traj_idxs: list[int],
    train_dt: float,
    display_name: str,
    out_path: Path,
    fit_cap: float = np.inf,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two-panel figure: max and median |ΔV| vs Δt, with per-trajectory power-law fits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    cmap = plt.get_cmap("tab10")

    panels = [
        ("max", max_step_dV, axes[0], r"$\max_k\, |V_\theta(u_{k+1}) - V_\theta(u_k)|$", "maximum"),
        ("median", median_step_dV, axes[1], r"$\mathrm{median}_k\, |V_\theta(u_{k+1}) - V_\theta(u_k)|$", "median"),
    ]

    all_slopes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stat_name, data, ax, ylabel, title in panels:
        slopes, intercepts, _, fit_masks, excluded_masks = _fit_powerlaw(dt_arr, data, finite, fit_cap)
        for k, ti in enumerate(traj_idxs):
            color = cmap(k % 10)
            fm = fit_masks[k]
            if fm.sum() >= 2:
                ax.plot(
                    dt_arr[fm],
                    data[fm, k],
                    "o",
                    color=color,
                    markersize=5,
                    label=rf"traj {ti}  ($\alpha$={slopes[k]:.2f})",
                )
                xs = dt_arr[fm]
                ax.plot(xs, np.exp(intercepts[k]) * xs ** slopes[k], "-", color=color, lw=1.0, alpha=0.6)
            em = excluded_masks[k]
            if em.any():
                ax.plot(dt_arr[em], data[em, k], "o", mfc="none", mec=color, markersize=5)

        ax.axvline(train_dt, color="gray", lw=0.8, linestyle=":", label=rf"train $\Delta t$={train_dt:.0e}")
        if np.isfinite(fit_cap):
            ax.axhline(fit_cap, color="gray", lw=0.8, linestyle="--", label=rf"fit cutoff $|\Delta V|<{fit_cap}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Inference step size $\Delta t$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(rf"{display_name} — {title} one-step $|\Delta V|$ vs $\Delta t$", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        all_slopes[stat_name] = (slopes, intercepts)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")
    return all_slopes


def plot_median_only(
    dt_arr: np.ndarray,
    median_step_dV: np.ndarray,
    finite: np.ndarray,
    traj_idxs: list[int],
    train_dt: float,
    out_path: Path,
    fit_cap: float = np.inf,
    extra_xticks: tuple[float, ...] = (5e-4, 1e-3, 5e-3, 1e-2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-panel median |ΔV| vs Δt plot, with R² in the legend."""
    palette = sns.color_palette("deep")
    slopes, intercepts, r2s, fit_masks, excluded_masks = _fit_powerlaw(dt_arr, median_step_dV, finite, fit_cap)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for k, _ti in enumerate(traj_idxs):
        color = palette[k % len(palette)]
        fm = fit_masks[k]
        if fm.sum() >= 2:
            ax.plot(
                dt_arr[fm],
                median_step_dV[fm, k],
                "o",
                color=color,
                markersize=5,
                label=rf"trajectory #{k + 1}  ($\alpha$={slopes[k]:.2f}, $R^2$={r2s[k]:.3f})",
            )
            xs = dt_arr[fm]
            ax.plot(xs, np.exp(intercepts[k]) * xs ** slopes[k], "-", color=color, lw=1.0, alpha=0.6)
        em = excluded_masks[k]
        if em.any():
            ax.plot(dt_arr[em], median_step_dV[em, k], "o", mfc="none", mec=color, markersize=5)

    ax.axvline(
        train_dt,
        color="gray",
        lw=0.8,
        linestyle="--",
        label=rf"time step $\Delta t$ used for training: ${_latex_sci(train_dt)}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Inference step size $\Delta t$", fontsize=14)
    ax.set_ylabel(
        r"$\mathrm{median}_s\, \left|V_\theta(u^{s+1}) - V_\theta(u^s)\right|$",
        fontsize=14,
    )
    ax.grid(True, which="both", alpha=0.3)

    ax.set_xlim(dt_arr.min() * 0.9, dt_arr.max() * 1.1)
    xt = sorted(set(extra_xticks))
    ax.xaxis.set_major_locator(FixedLocator(xt))

    def _fmt_dt(x, _pos):
        if np.isclose(x, 5e-3):
            return r"$5\times10^{-3}$"
        if np.isclose(x, 5e-4):
            return r"$5\times10^{-4}$"
        e = int(round(np.log10(x)))
        return rf"$10^{{{e}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_dt))
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")
    return slopes, intercepts, r2s


# ---------------------------------------------------------------------------
# Allen-Cahn / Allen-Cahn-2D trajectory sweep — V(u^t) and rel-RMSE vs lead time
# ---------------------------------------------------------------------------


def make_dt_factor_plan(
    train_dt: float,
    T_final: float,
    factors: tuple = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
) -> list[tuple[float, int]]:
    """Δt grid given as multiples of train_dt — n_steps = T_final / Δt rounded."""
    return [(float(train_dt * f), max(1, int(round(T_final / (train_dt * f))))) for f in factors]


def run_dt_trajectory_sweep(
    model: MesoLitModule,
    test_data: torch.Tensor,
    traj_idxs: list[int],
    plan: list[tuple[float, int]],
    dt_truth: float,
    device: str,
    verbose: bool = True,
    v_blowup_factor: float = 50.0,
    compute_V: bool | None = None,
) -> dict[float, dict]:
    """Roll model out at each Δt and record V(u^t) and rel-RMSE on the truth grid.

    For Δt that is a clean multiple/divisor of `dt_truth`, predicted states line
    up with truth indices; rel-RMSE is recorded at those matched times. Once a
    trajectory diverges (non-finite, or |V| exceeds `v_blowup_factor` × |V(0)|),
    its V/rel-RMSE entries are set to NaN from that step onward and the state
    is reset to u0 to keep forward_step well-defined for the remaining trajs.

    Returns ``{dt: {"t": (S+1,), "V": (S+1, N_traj),
                    "t_rmse": (M,), "rel_rmse": (M, N_traj)}}``.

    If the model has no `dynamics.potential` (e.g. FNO baseline) or
    `compute_V=False`, V entries are filled with NaN and divergence is detected
    only via non-finite states.
    """
    if compute_V is None:
        compute_V = hasattr(model.dynamics, "potential")
    potential = model.dynamics.potential if compute_V else None
    u0_batch = test_data[traj_idxs, 0].to(device)
    truth = test_data[traj_idxs].to(device)  # (N_traj, T, n_vars, *spatial)
    N_traj = u0_batch.shape[0]
    T_data = truth.shape[1]

    out: dict[float, dict] = {}

    with torch.no_grad():
        for dt, n_steps in plan:
            x = u0_batch.clone()
            t_arr = np.arange(n_steps + 1, dtype=np.float64) * dt
            V_arr = np.full((n_steps + 1, N_traj), np.nan, dtype=np.float64)
            if compute_V:
                V0 = _compute_V(potential, x).cpu().numpy()
                V_arr[0] = V0
                v_thresh = v_blowup_factor * np.maximum(np.abs(V0), 1.0)
            else:
                v_thresh = None

            rmse_times: list[float] = [0.0]
            rmse_vals: list[np.ndarray] = [np.zeros(N_traj)]

            diverged = np.zeros(N_traj, dtype=bool)
            reduce_dims = tuple(range(1, x.ndim))
            shape_for_mask = (N_traj,) + (1,) * (x.ndim - 1)

            for s in range(1, n_steps + 1):
                x = model.dynamics.forward_step(x, dt=dt)
                finite_now = torch.isfinite(x).reshape(N_traj, -1).all(dim=1).cpu().numpy()
                if compute_V:
                    V_now = _compute_V(potential, x).cpu().numpy()
                    blow = (np.abs(V_now) > v_thresh) | (~finite_now)
                else:
                    V_now = None
                    blow = ~finite_now
                diverged = diverged | blow

                if blow.any():
                    bad = torch.from_numpy(blow).to(device)
                    x = torch.where(bad.view(*shape_for_mask), u0_batch, x)

                if compute_V:
                    V_arr[s] = np.where(diverged, np.nan, V_now)

                t_now = s * dt
                idx_f = t_now / dt_truth
                idx_lo = int(np.floor(idx_f))
                idx_hi = idx_lo + 1
                if idx_lo < 0 or idx_lo >= T_data:
                    truth_at_t = None
                elif idx_hi >= T_data:
                    truth_at_t = truth[:, idx_lo] if abs(idx_f - idx_lo) < 1e-9 else None
                else:
                    alpha = idx_f - idx_lo
                    truth_at_t = (1.0 - alpha) * truth[:, idx_lo] + alpha * truth[:, idx_hi]
                if truth_at_t is not None:
                    diff = x - truth_at_t
                    ref = (truth_at_t**2).mean(dim=reduce_dims) + 1e-12
                    rel = torch.sqrt((diff**2).mean(dim=reduce_dims) / ref).cpu().numpy()
                    rel = np.where(diverged, np.nan, rel)
                    rmse_times.append(t_now)
                    rmse_vals.append(rel)

            out[float(dt)] = {
                "t": t_arr,
                "V": V_arr,
                "t_rmse": np.array(rmse_times),
                "rel_rmse": np.stack(rmse_vals),
            }
            if verbose:
                final_rmse = rmse_vals[-1]
                n_div = int(diverged.sum())
                v_part = f"V(0)={V_arr[0].mean():.4f}  V(T)={np.nanmean(V_arr[-1]):.4f}  " if compute_V else ""
                print(
                    f"Δt={dt:.2e}  n_steps={n_steps:>5d}  "
                    + v_part
                    + f"final rel_rmse={np.nanmean(final_rmse):.4f}"
                    + (f"  ({n_div}/{N_traj} traj diverged)" if n_div else "")
                )

    return out


def _dt_colors(dts: list[float], cmap_name: str = "Blues") -> list:
    """Return one color per Δt — light = small Δt, dark = large Δt."""
    cmap = plt.get_cmap(cmap_name)
    n = len(dts)
    if n == 1:
        return [cmap(0.7)]
    return [cmap(0.25 + 0.7 * i / (n - 1)) for i in range(n)]


def _dt_label(dt: float, train_dt: float) -> str:
    base = rf"$\Delta t={_latex_sci(dt)}$"
    return base + (r" (train)" if abs(dt - train_dt) < 1e-12 * max(1.0, train_dt) else "")


def plot_potential_vs_lead_time(
    sweep: dict[float, dict],
    train_dt: float,
    display_name: str,
    out_path: Path,
    cmap_name: str = "Blues",
    margin_frac: float = 0.1,
) -> None:
    """V(u^t) vs prediction lead time, one curve per Δt (mean ± std over trajs).

    The y-axis is set to the train-Δt trajectory's V-range expanded by
    ``margin_frac`` on top and bottom — so divergent rollouts at larger Δt
    are clipped instead of crushing the visible curves.
    """
    dts = sorted(sweep.keys())
    colors = _dt_colors(dts, cmap_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    import warnings

    train_key = min(sweep.keys(), key=lambda d: abs(d - train_dt))
    V_train_mean = np.nanmean(sweep[train_key]["V"], axis=1)
    v_min = float(np.nanmin(V_train_mean))
    v_max = float(np.nanmax(V_train_mean))
    v_range = v_max - v_min
    ylim_bottom = v_min - margin_frac * v_range
    ylim_top = v_max + margin_frac * v_range

    for dt, color in zip(dts, colors):
        r = sweep[dt]
        V = r["V"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(V, axis=1)
            s = np.nanstd(V, axis=1)
        ax.plot(r["t"], m, color=color, lw=1.6, label=_dt_label(dt, train_dt))
        ax.fill_between(r["t"], m - s, m + s, color=color, alpha=0.15, lw=0)

    ax.set_ylim(ylim_bottom, ylim_top)

    ax.set_xlabel(r"Prediction lead time $t$", fontsize=14)
    ax.set_ylabel(r"$V_\theta(u(t))$", fontsize=14)
    # ax.set_title(rf"{display_name} — learned potential along rollout vs $\Delta t$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


def plot_rel_rmse_vs_lead_time(
    sweep: dict[float, dict],
    train_dt: float,
    display_name: str,
    out_path: Path,
    cmap_name: str = "Blues",
    yscale: str = "log",
    ylim: tuple[float, float] | None = (1e-3, 1e-1),
) -> None:
    """Relative RMSE (vs ground truth) vs prediction lead time, one curve per Δt."""
    dts = sorted(sweep.keys())
    colors = _dt_colors(dts, cmap_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    import warnings

    for dt, color in zip(dts, colors):
        r = sweep[dt]
        # skip the t=0 zero entry on log scale
        t = r["t_rmse"]
        rmse = r["rel_rmse"]
        if yscale == "log":
            mask = t > 0
            t = t[mask]
            rmse = rmse[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(rmse, axis=1)
            s = np.nanstd(rmse, axis=1)
        ax.plot(t, m, color=color, lw=1.6, label=_dt_label(dt, train_dt))
        ax.fill_between(t, np.maximum(m - s, 1e-12), m + s, color=color, alpha=0.15, lw=0)

    ax.set_xlabel(r"Prediction lead time $t$", fontsize=14)
    ax.set_ylabel(r"Relative RMSE", fontsize=14)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.set_title(rf"{display_name} — prediction error along rollout vs $\Delta t$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


def plot_final_rel_rmse_vs_dt(
    sweep: dict[float, dict],
    train_dt: float,
    display_name: str,
    out_path: Path,
) -> None:
    """Final-time relative RMSE (mean ± std over trajs) vs Δt."""
    import warnings

    dts = np.array(sorted(sweep.keys()), dtype=np.float64)
    means = np.full(len(dts), np.nan)
    stds = np.full(len(dts), np.nan)
    for i, dt in enumerate(dts):
        rel = sweep[float(dt)]["rel_rmse"][-1]  # final-time slice over trajs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means[i] = np.nanmean(rel)
            stds[i] = np.nanstd(rel)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(dts, means, yerr=stds, fmt="o-", color="tab:blue", lw=1.6, capsize=3)
    ax.axvline(
        train_dt,
        color="gray",
        lw=0.8,
        linestyle="--",
        label=rf"train $\Delta t = {_latex_sci(train_dt)}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Inference step size $\Delta t$", fontsize=12)
    ax.set_ylabel(r"Final-time relative RMSE", fontsize=12)
    ax.set_title(rf"{display_name} — final relative RMSE vs $\Delta t$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")


def plot_final_rel_rmse_vs_dt_overlay(
    sweeps_by_label: dict[str, dict[float, dict]],
    train_dt: float,
    display_name: str,
    out_path: Path,
    colors: dict[str, object] | None = None,
) -> None:
    """Final-time relative RMSE (mean ± std over trajs) vs Δt, overlaying models."""
    import warnings

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette("deep")
    for i, (label, sweep) in enumerate(sweeps_by_label.items()):
        dts = np.array(sorted(sweep.keys()), dtype=np.float64)
        means = np.full(len(dts), np.nan)
        stds = np.full(len(dts), np.nan)
        for j, dt in enumerate(dts):
            rel = sweep[float(dt)]["rel_rmse"][-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means[j] = np.nanmean(rel)
                stds[j] = np.nanstd(rel)
        color = (colors or {}).get(label, palette[i % len(palette)])
        ax.errorbar(dts, means, yerr=stds, fmt="o-", color=color, lw=1.6, capsize=3, label=label)

    ax.axvline(
        train_dt,
        color="gray",
        lw=0.8,
        linestyle="--",
        label=rf"train $\Delta t = {_latex_sci(train_dt)}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Inference step size $\Delta t$", fontsize=12)
    ax.set_ylabel(r"Final-time relative RMSE", fontsize=12)
    ax.set_title(rf"{display_name} — final relative RMSE vs $\Delta t$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")
