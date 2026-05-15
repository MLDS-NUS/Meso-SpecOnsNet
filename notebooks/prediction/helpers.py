"""Shared helpers for the prediction notebooks (ac / kdv).

Each notebook only needs to specify its dataset, run directory, and plot
options — model loading, rollout/RMSE computation, and plotting are done here.
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
import torch
import yaml
from omegaconf import OmegaConf

from src.models.meso_module import MesoLitModule


def find_latest_ckpt(run_dir: Path, use_final_loss: bool = False) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        if use_final_loss:
            best_final = ckpt_dir / "best_final_loss.ckpt"
            if best_final.exists():
                return best_final
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


def load_model_for_inference(
    run_dir: Path,
    root: Path,
    device: str = "cpu",
    use_final_loss: bool = False,
) -> MesoLitModule:
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

    ckpt_path = find_latest_ckpt(run_dir, use_final_loss=use_final_loss)
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
    u_test = u_all[val_end:]
    u_test = np.transpose(u_test, (0, 1, 3, 2))  # (N_test, T, n_vars, Nx)
    return torch.from_numpy(u_test).float(), t_coord, x_coord


def load_test_data_nd(
    data_glob: str,
    n_spatial_dims: int = 1,
    split_ratio: tuple = (0.7, 0.2, 0.1),
) -> tuple:
    """N-D variant of load_test_data.

    Returns (u_test, t_coord, x_coord, y_coord). y_coord is None for 1D data.
    Output tensor has shape (N_test, T, n_vars, *spatial).
    """
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


def discover_models(
    runs_dir: Path,
    root: Path,
    device: str = "cpu",
    use_final_loss: bool = False,
) -> dict:
    models: dict = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        exp_name = run_dir.name
        print(f"Loading '{exp_name}' ...")
        try:
            m = load_model_for_inference(run_dir, root=root, device=device, use_final_loss=use_final_loss)
            models[exp_name] = {"model": m, "label": exp_name}
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\n{len(models)} model(s) loaded: {list(models.keys())}")
    return models


def compute_rel_rmse_rollout(
    model: MesoLitModule,
    eval_data: torch.Tensor,
    max_steps: int,
    device: str,
) -> np.ndarray:
    """Autoregressive rollout, returning per-sample relative RMSE per step.

    Once any sample first exceeds rel_rmse >= 1 at step t, all samples from
    step t onward are clamped to 1.0 so the mean jumps above the y-limit.
    """
    N_eval = eval_data.shape[0]
    batch = eval_data.to(device)
    truth = batch.cpu().numpy()
    # average over (n_vars, *spatial) — leaves shape (N, T)
    ref_norm = (truth**2).mean(axis=tuple(range(2, truth.ndim))) + 1e-12

    rel_rmse_ps = np.zeros((N_eval, max_steps + 1))
    x = batch[:, 0].clone()
    with torch.no_grad():
        for t in range(1, max_steps + 1):
            x = model.dynamics.forward_step(x, dt=model.dt)
            pred_np = x.cpu().numpy()
            diff = pred_np - truth[:, t]
            # average over (n_vars, *spatial) — leaves shape (N,)
            diff_axes = tuple(range(1, diff.ndim))
            rel_rmse_ps[:, t] = np.sqrt((diff**2).mean(axis=diff_axes) / ref_norm[:, t])

    diverge_steps = np.where((rel_rmse_ps >= 1.0).any(axis=0))[0]
    if len(diverge_steps):
        first_div = diverge_steps[0]
        rel_rmse_ps[:, first_div:] = 1.0
        print(f"  diverges at step {first_div} (any sample >= 1) -> set to 1.0 from there")
    else:
        print(f"  no divergence within {max_steps} steps")
    return rel_rmse_ps


def run_all_rollouts(
    models: dict,
    eval_data: torch.Tensor,
    max_steps: int,
    device: str,
) -> dict:
    results: dict = {}
    for exp_name, m_info in models.items():
        print(f"Rollout for '{exp_name}'...")
        rel_rmse = compute_rel_rmse_rollout(m_info["model"], eval_data, max_steps, device)
        results[exp_name] = {"rel_rmse": rel_rmse}
    print("\nAll rollouts complete.")
    return results


def print_summary_table(
    results: dict,
    display_names: dict,
    report_steps: list,
    max_steps: int,
) -> None:
    report_steps = [s for s in report_steps if s <= max_steps]
    col_w = 22
    label_w = max((len(display_names.get(k, k)) for k in results), default=20) + 2
    header = f"{'Model':<{label_w}}" + "".join(f"{'Step ' + str(s):>{col_w}}" for s in report_steps)
    print(header)
    print("-" * len(header))
    for exp_name, res in results.items():
        label = display_names.get(exp_name, exp_name)
        row = f"{label:<{label_w}}"
        for s in report_steps:
            vals = res["rel_rmse"][:, s]
            row += f"{vals.mean():.4f} ± {vals.std():.4f}".rjust(col_w)
        print(row)


def plot_rel_rmse(
    results: dict,
    plot_models: list,
    display_names: dict,
    model_colors: dict,
    out_path: Path,
    max_steps: int,
    ylim: tuple = (1e-3, 1.0),
) -> None:
    plot_items = [(k, results[k]) for k in plot_models if k in results]
    steps = np.arange(max_steps + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for exp_name, res in plot_items:
        color = model_colors.get(exp_name)
        data = res["rel_rmse"]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        label = display_names.get(exp_name, exp_name)
        ax.plot(steps[1:], mean[1:], label=label, color=color, linewidth=2)
        ax.fill_between(
            steps[1:],
            np.maximum(mean[1:] - std[1:], 1e-12),
            mean[1:] + std[1:],
            color=color,
            alpha=0.2,
        )

    ax.set_yscale("log")
    ax.set_xlim(1, max_steps)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_xlabel("Prediction step", fontsize=12)
    ax.set_ylabel("Relative RMSE", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {out_path}")
