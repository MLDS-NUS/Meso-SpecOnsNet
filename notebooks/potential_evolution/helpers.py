"""Shared helpers for the potential-evolution notebooks (kdv / fpu / fene).

Each notebook only needs to specify its dataset glob, run directory, and output
name — the loading, V computation, and plotting are done here.
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
from omegaconf import OmegaConf

from src.models.meso_module import MesoLitModule


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
    u_test = u_all[val_end:]
    u_test = np.transpose(u_test, (0, 1, 3, 2))  # (N_test, T, n_vars, Nx)
    return torch.from_numpy(u_test).float(), t_coord, x_coord


def compute_V_learned(potential, test_data: torch.Tensor, device: str) -> np.ndarray:
    """Compute V_theta(u(t)) = V0 + V1 + V2 for each trajectory."""
    N_test, T = test_data.shape[0], test_data.shape[1]
    V = np.zeros((N_test, T), dtype=np.float64)
    with torch.no_grad():
        for i, u_traj in enumerate(test_data):
            u_traj = u_traj.to(device)
            _V0 = potential._V0(u_traj)
            _V1 = potential._V1(u_traj)
            _V2 = potential._V2(u_traj).view(T, 1)
            V[i] = (_V0 + _V1 + _V2).squeeze(-1).cpu().numpy()
    return V


def _make_colors(n_plot: int):
    return sns.color_palette("deep", n_colors=n_plot)


def plot_V_evolution(V_learned: np.ndarray, out_path: Path, n_plot: int = 10) -> None:
    N_test, T = V_learned.shape
    traj_idx = np.linspace(0, N_test - 1, n_plot, dtype=int)
    steps = np.arange(T)
    colors = _make_colors(n_plot)

    fig, ax = plt.subplots(figsize=(6, 4))
    for plot_i, traj_i in enumerate(traj_idx):
        ax.plot(steps, V_learned[traj_i], color=colors[plot_i], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Prediction step", fontsize=14)
    ax.set_ylabel(r"Learned potential $V_\theta(u(t))$", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


def plot_V_change(V_learned: np.ndarray, out_path: Path, n_plot: int = 10) -> None:
    N_test, T = V_learned.shape
    traj_idx = np.linspace(0, N_test - 1, n_plot, dtype=int)
    steps = np.arange(T)
    colors = _make_colors(n_plot)

    fig, ax = plt.subplots(figsize=(7, 4))
    for plot_i, traj_i in enumerate(traj_idx):
        vl = V_learned[traj_i]
        ax.plot(steps, vl - vl[0], color=colors[plot_i], linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Prediction step", fontsize=14)
    ax.set_ylabel(r"$V_\theta(u(t)) - V_\theta(u(0))$", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
