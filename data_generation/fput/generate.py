r"""Generate dataset of 1D chain (FPU lattice) dynamics mapping to KdV equation.

Chain model: quadratic nonlinear 1D lattice (Fermi-Pasta-Ulam type).
  Force law:    F(r) = c^2 * r + alpha * r^2
  Strain EOM:   r_ddot_n = c^2*(r_{n+1}-2r_n+r_{n-1}) + alpha*(r_{n+1}^2-2r_n^2+r_{n-1}^2)

Maps to KdV in the long-wave small-amplitude limit via multiscale expansion:
  u_tau + a_kdv * u * u_xi + b_kdv * u_xixixi = 0

Scaling ansatz (see CHAIN_KDV_MAPPING.md for derivation):
  r_n(t_lab) = eps^2 * u(xi, tau)
  xi  = eps * (n - c * t_lab)      [co-moving slow space]
  tau = eps^3 * t_lab              [slow time]

Parameter mapping:
  c     = 24 * b_kdv
  alpha = a_kdv * c
  eps   = Lx / N_chain             [small parameter, set by grid choice]

Output: u(xi, tau) reconstructed from chain strains on a regular KdV grid.
Uses Störmer-Verlet (leapfrog) time integration for symplectic accuracy.

Stability note: CFL condition requires c * dt_chain < 1 (lattice units).
  dt_chain = dt_kdv / (eps^3 * chain_substeps)
  Increase chain_substeps if the simulation blows up.
"""

import argparse
import logging
import os
import sys
import time

import h5py
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# ─── Initial condition ─────────────────────────────────────────────────────────


class ChainKdVIC:
    """Random smooth periodic initial condition u0(xi) for the KdV field.

    Generates u0 as a superposition of Nk sine waves with random wavenumbers,
    amplitudes, and phases, normalized to unit standard deviation.
    """

    def __init__(self, Lx: float, Nk: int, max_k: int) -> None:
        self.Lx = Lx
        self.Nk = Nk
        self.max_k = max_k

    def reset(self, rng: np.random.Generator) -> None:
        """Draw new random IC parameters."""
        self.ks = rng.choice(np.arange(1, self.max_k + 1), size=self.Nk, replace=False)
        self.amps = rng.uniform(0.0, 1.0, size=self.Nk)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.Nk)
        # Normalize to unit std using a dense reference grid
        x_ref = np.linspace(0.0, self.Lx, 1024, endpoint=False)
        std = np.std(self._raw_eval(x_ref))
        self.norm = std if std > 1e-12 else 1.0

    def _raw_eval(self, x: np.ndarray) -> np.ndarray:
        field = np.zeros_like(x, dtype=float)
        for k, a, phi in zip(self.ks, self.amps, self.phases):
            field += a * np.sin(2.0 * np.pi * k * x / self.Lx + phi)
        return field

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate u0(x) at arbitrary points (normalized)."""
        return self._raw_eval(x) / self.norm

    def eval_deriv(self, x: np.ndarray) -> np.ndarray:
        """Evaluate u0'(x) at arbitrary points (normalized)."""
        field = np.zeros_like(x, dtype=float)
        for k, a, phi in zip(self.ks, self.amps, self.phases):
            field += a * (2.0 * np.pi * k / self.Lx) * np.cos(2.0 * np.pi * k * x / self.Lx + phi)
        return field / self.norm

    def get_data_dict(self, prefix: str) -> dict:
        return {
            prefix + "/ks": self.ks,
            prefix + "/amps": self.amps,
            prefix + "/phases": self.phases,
        }

    def __str__(self) -> str:
        if not hasattr(self, "ks"):
            return "ChainKdVIC(uninitialized)"
        return f"ChainKdVIC(ks={self.ks}, amps={self.amps.round(3)})"


# ─── Chain physics ─────────────────────────────────────────────────────────────


def _chain_acc(r: np.ndarray, c2: float, alpha: float) -> np.ndarray:
    """Strain acceleration for periodic chain.

    acc_n = c^2*(r_{n+1} - 2*r_n + r_{n-1}) + alpha*(r_{n+1}^2 - 2*r_n^2 + r_{n-1}^2)
    """
    r_fwd = np.roll(r, -1)  # r_{n+1}
    r_bwd = np.roll(r, 1)  # r_{n-1}
    return c2 * (r_fwd - 2.0 * r + r_bwd) + alpha * (r_fwd**2 - 2.0 * r**2 + r_bwd**2)


def _periodic_interp(xi_src: np.ndarray, u_src: np.ndarray, xi_dst: np.ndarray, Lx: float) -> np.ndarray:
    """Linear interpolation of periodic data onto a regular grid.

    Wraps xi_src into [0, Lx), sorts, and extends periodically before
    calling np.interp so that the result is valid everywhere in xi_dst.
    """
    xi_w = xi_src % Lx
    idx = np.argsort(xi_w)
    xs = xi_w[idx]
    us = u_src[idx]
    # Periodic extension: prepend and append one copy
    xs_ext = np.concatenate([xs - Lx, xs, xs + Lx])
    us_ext = np.concatenate([us, us, us])
    return np.interp(xi_dst, xs_ext, us_ext)


# ─── Core simulation ───────────────────────────────────────────────────────────


def simulate_chain_kdv(
    u_ic: ChainKdVIC,
    *,
    a_kdv: float,
    b_kdv: float,
    eps: float,
    Lx: float,
    N_chain: int,
    N_kdv: int,
    dt_kdv: float,
    T: float,
    chain_substeps: int,
    save_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the chain and return u(xi, tau) on the KdV grid.

    Chain is integrated in lab time using Störmer-Verlet (leapfrog).
    At each saved slow-time step, strains are rescaled and interpolated
    onto the co-moving KdV grid.

    Returns:
        t_slow:   shape (n_frames,), slow time tau = eps^3 * t_lab
        xi_grid:  shape (N_kdv,), co-moving spatial coordinate
        u_series: shape (n_frames, N_kdv), reconstructed KdV field
    """
    # Derived chain parameters from KdV coefficients
    c = 24.0 * b_kdv
    alpha = a_kdv * c
    c2 = c**2
    dt_chain = dt_kdv / (eps**3 * chain_substeps)

    # Chain site indices and their initial slow-frame positions
    n_sites = np.arange(N_chain, dtype=float)
    xi_n0 = eps * n_sites  # xi at t_lab=0

    # KdV output grid
    xi_grid = np.linspace(0.0, Lx, N_kdv, endpoint=False)

    # Chain initial conditions from the KdV IC via the scaling ansatz:
    #   r_n(0) = eps^2 * u0(eps*n)
    #   rdot_n(0) = -c * eps^3 * u0'(eps*n)   [right-traveling wave IC]
    r = eps**2 * u_ic.eval(xi_n0)
    v = -c * eps**3 * u_ic.eval_deriv(xi_n0)

    total_kdv_steps = round(T / dt_kdv)
    n_frames = total_kdv_steps // save_every + 1
    u_series = np.zeros((n_frames, N_kdv), dtype=np.float32)
    t_slow = np.zeros(n_frames, dtype=np.float64)

    def reconstruct_u(r: np.ndarray, t_lab: float) -> np.ndarray:
        # Slow-frame coordinate of each chain site at current t_lab
        xi_n = eps * n_sites - eps * c * t_lab  # wraps via % Lx inside interp
        return _periodic_interp(xi_n, r / eps**2, xi_grid, Lx)

    # Record t=0
    t_lab = 0.0
    u_series[0] = reconstruct_u(r, t_lab)
    frame_idx = 1

    for step in range(1, total_kdv_steps + 1):
        # Inner loop: chain_substeps of Störmer-Verlet
        for _ in range(chain_substeps):
            acc = _chain_acc(r, c2, alpha)
            v_half = v + 0.5 * dt_chain * acc
            r = r + dt_chain * v_half
            acc_new = _chain_acc(r, c2, alpha)
            v = v_half + 0.5 * dt_chain * acc_new
            t_lab += dt_chain

        if step % save_every == 0 and frame_idx < n_frames:
            u_series[frame_idx] = reconstruct_u(r, t_lab)
            t_slow[frame_idx] = eps**3 * t_lab
            frame_idx += 1

    return t_slow[:frame_idx], xi_grid, u_series[:frame_idx]


# ─── Data generator ────────────────────────────────────────────────────────────


class ChainKdVDataGen:
    r"""Generate dataset of 1D chain dynamics in the KdV co-moving frame.

    ======== Chain-KdV system ========
    Fermi-Pasta-Ulam chain with F(r) = c^2*r + alpha*r^2, periodic BC.
    Long-wave small-amplitude limit gives KdV:
      u_tau + a_kdv * u * u_xi + b_kdv * u_xixixi = 0
    Output: u(xi, tau) reconstructed on a co-moving KdV spatial grid.
    """

    info_dict = {"version": 1.0, "preprocess_dag": False, "pde_type_id": 3001}

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.a_kdv = args.a_kdv
        self.b_kdv = args.b_kdv
        self.Lx = args.Lx
        self.N_chain = args.N_chain
        self.eps = args.Lx / args.N_chain  # small parameter
        self.N_kdv = args.N_kdv
        self.dt_kdv = args.dt_kdv
        self.T = args.T
        self.chain_substeps = args.chain_substeps
        self.save_every = args.save_every
        self.num_samples = args.num_samples
        self.u_bound = args.u_bound

        self.u_ic = ChainKdVIC(args.Lx, args.Nk, args.max_k)

        self.t_coord: np.ndarray | None = None
        self.x_coord: np.ndarray | None = None
        self.u_sol_all: list[np.ndarray] = []
        self.coef_ks: list[np.ndarray] = []
        self.coef_amps: list[np.ndarray] = []
        self.coef_phases: list[np.ndarray] = []

    # ── Filename ───────────────────────────────────────────────────────────────

    def get_file_stem(self) -> str:
        args = self.args
        version = self.info_dict["version"]
        stem = f"custom_v{version:g}_fput_nx{self.N_kdv}_nchain{self.N_chain}"
        if args.num_samples != 10000:
            stem += f"_num{args.num_samples}"
        if args.np_seed == -1:
            stem += time.strftime("_%Y-%m-%d-%H-%M-%S")
        else:
            stem += f"_seed{args.np_seed}"
        return stem

    # ── Per-sample ─────────────────────────────────────────────────────────────

    def reset(self, rng: np.random.Generator) -> None:
        self.u_ic.reset(rng)

    def run_simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return simulate_chain_kdv(
            self.u_ic,
            a_kdv=self.a_kdv,
            b_kdv=self.b_kdv,
            eps=self.eps,
            Lx=self.Lx,
            N_chain=self.N_chain,
            N_kdv=self.N_kdv,
            dt_kdv=self.dt_kdv,
            T=self.T,
            chain_substeps=self.chain_substeps,
            save_every=self.save_every,
        )

    def _accept(self, u_series: np.ndarray, log_fn=None) -> bool:
        if not np.isfinite(u_series).all():
            if callable(log_fn):
                log_fn("rejected: non-finite values")
            return False
        u_max = float(np.max(np.abs(u_series)))
        if u_max > self.u_bound:
            if callable(log_fn):
                log_fn(f"rejected: u_max={u_max:.2f} > {self.u_bound}")
            return False
        return True

    def _record(self, t_slow: np.ndarray, xi_grid: np.ndarray, u_series: np.ndarray) -> None:
        if self.t_coord is None:
            self.t_coord = t_slow
            self.x_coord = xi_grid
        self.u_sol_all.append(u_series.astype(np.float32))
        coef = self.u_ic.get_data_dict("u_ic")
        self.coef_ks.append(coef["u_ic/ks"])
        self.coef_amps.append(coef["u_ic/amps"])
        self.coef_phases.append(coef["u_ic/phases"])

    # ── Main loop ──────────────────────────────────────────────────────────────

    def gen_data_all(self, rng: np.random.Generator, log_fn=None) -> None:
        trial = 0
        while len(self.u_sol_all) < self.num_samples:
            trial += 1
            if callable(log_fn):
                log_fn(f"trial {trial}, collected {len(self.u_sol_all)}/{self.num_samples}  IC: {self.u_ic}")
            self.reset(rng)
            if callable(log_fn):
                log_fn(f"  IC: {self.u_ic}")
            t_slow, xi_grid, u_series = self.run_simulation()
            if self._accept(u_series, log_fn):
                self._record(t_slow, xi_grid, u_series)

    # ── Save ───────────────────────────────────────────────────────────────────

    def save_hdf5(self) -> str:
        args = self.args
        os.makedirs(args.h5_file_dir, exist_ok=True)
        stem = self.get_file_stem()
        path = os.path.join(args.h5_file_dir, stem + ".hdf5")

        c = 24.0 * self.b_kdv
        alpha = self.a_kdv * c

        with h5py.File(path, "w") as f:
            f.create_dataset("t_coord", data=self.t_coord)
            f.create_dataset("x_coord", data=self.x_coord)
            # u_sol_all: (num_samples, n_frames, N_kdv, 1) — nvar dim appended for consistency
            f.create_dataset("u_sol_all", data=np.array(self.u_sol_all)[..., None])
            f.create_dataset("coef/u_ic/ks_all", data=np.array(self.coef_ks))
            f.create_dataset("coef/u_ic/amps_all", data=np.array(self.coef_amps))
            f.create_dataset("coef/u_ic/phases_all", data=np.array(self.coef_phases))
            # Chain-KdV physics parameters
            f.create_dataset("pde_info/a_kdv", data=self.a_kdv)
            f.create_dataset("pde_info/b_kdv", data=self.b_kdv)
            f.create_dataset("pde_info/eps", data=self.eps)
            f.create_dataset("pde_info/c", data=c)
            f.create_dataset("pde_info/alpha", data=alpha)
            for key, value in self.info_dict.items():
                f.create_dataset(f"pde_info/{key}", data=value)
            for key, value in vars(args).items():
                if isinstance(value, int | float | np.ndarray):
                    f.create_dataset(f"pde_info/args/{key}", data=value)

        return path


# ─── CLI ───────────────────────────────────────────────────────────────────────


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ChainKdVDataGen.__doc__)
    parser.add_argument("--num_samples", "-n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--np_seed", "-r", type=int, default=-1, help="NumPy random seed (-1 = random)")
    parser.add_argument("--u_bound", type=float, default=20.0, help="Reject samples with max|u| > u_bound")
    # KdV parameters
    parser.add_argument("--a_kdv", type=float, default=1.0, help="KdV nonlinear coefficient")
    parser.add_argument("--b_kdv", type=float, default=1.0 / 24.0, help="KdV dispersion coefficient")
    # Domain
    parser.add_argument("--Lx", type=float, default=2.0 * np.pi, help="Periodic domain length")
    parser.add_argument("--N_kdv", type=int, default=256, help="KdV output grid points")
    parser.add_argument("--N_chain", type=int, default=128, help="Number of chain sites (eps = Lx/N_chain)")
    # Time
    parser.add_argument("--dt_kdv", type=float, default=1e-3, help="Slow-time step size (KdV units)")
    parser.add_argument("--T", type=float, default=1.0, help="Total slow time")
    parser.add_argument("--chain_substeps", type=int, default=50, help="Lab-time substeps per slow-time step")
    parser.add_argument("--save_every", type=int, default=10, help="Save u every this many slow-time steps")
    # IC
    parser.add_argument("--Nk", type=int, default=3, help="Number of Fourier modes in IC")
    parser.add_argument("--max_k", type=int, default=6, help="Max Fourier wavenumber in IC")
    # Output
    parser.add_argument("--h5_file_dir", type=str, default="data/fput")
    return parser.parse_args()


def _create_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("chain_kdv")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    args = _get_args()

    gen = ChainKdVDataGen(args)

    os.makedirs("log", exist_ok=True)
    logger = _create_logger(os.path.join("log", gen.get_file_stem() + ".log"))
    logger.info("target file: %s.hdf5", gen.get_file_stem())

    eps = gen.eps
    c = 24.0 * args.b_kdv
    alpha = args.a_kdv * c
    dt_chain = args.dt_kdv / (eps**3 * args.chain_substeps)
    logger.info("eps = %.5f  (N_chain=%d, Lx=%.4f)", eps, args.N_chain, args.Lx)
    logger.info("c = %.4f,  alpha = %.4f", c, alpha)
    logger.info(
        "dt_chain = %.3e  (CFL = c*dt_chain = %.3f, should be < 1)",
        dt_chain,
        c * dt_chain,
    )

    rng = np.random.default_rng(None if args.np_seed == -1 else args.np_seed)
    gen.gen_data_all(rng, log_fn=logger.info)
    path = gen.save_hdf5()
    logger.info("file saved: %s", path)
