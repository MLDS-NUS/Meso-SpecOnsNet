r"""Generate dataset of 1D FENE-v2 chain dynamics.

FENE-v2 chain model: displacement formulation of a FENE lattice.
  Potential:    V(r) = -H*R^2/2 * log(1 - r^2/R^2)
  Force law:    F(r) = H*r / (1 - (r/R)^2)
  Strain:       r_n = q_{n+1} - q_n
  EOM:          q_ddot_n = F(r_n) - F(r_{n-1})

Parameters follow the KdV chain scaling convention:
  c  = linear wave speed,   H = c^2
  R  = eps^2 * 50           (finite extensibility limit)
  eps = Lx / N_chain        (small parameter, set by grid choice)

Scaling ansatz for IC and output (co-moving frame):
  r_n(t_lab) = eps^2 * u(xi, tau)
  xi  = eps * (n - c * t_lab)   [co-moving slow space]
  tau = eps^3 * t_lab            [slow time]

Output: u(xi, tau) = strain(q)/eps^2 reconstructed in the co-moving frame
on a regular spatial grid, same format as chain_kdv.

Uses Störmer-Verlet (leapfrog) time integration for symplectic accuracy.

Stability note: CFL condition requires c * dt_chain < 1 (lattice units).
  dt_chain = dt_kdv / (eps^3 * chain_substeps)
  Increase chain_substeps if the simulation blows up.
FENE note: strains must satisfy |r_n| < R. Samples violating this are rejected.
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


class FENEv2IC:
    """Random smooth periodic initial condition u0(xi) for the slow field.

    Generates u0 as a superposition of Nk sine waves with random wavenumbers,
    amplitudes, and phases, normalized to unit standard deviation.
    Chain IC is built via the scaling ansatz:
      r_n(0) = eps^2 * u0(eps*n)
      v_n(0) = -c * eps^3 * u0'(eps*n)
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
            return "FENEv2IC(uninitialized)"
        return f"FENEv2IC(ks={self.ks}, amps={self.amps.round(3)})"


# ─── FENE-v2 physics ───────────────────────────────────────────────────────────


def _fene_v2_acc(q: np.ndarray, H: float, R: float) -> np.ndarray:
    """Displacement acceleration for periodic FENE-v2 chain.

    Equivalent to the prototype:
        def acceleration(q):
            r = strain(q)         # r_n = q_{n+1} - q_n
            H = c**2
            R = eps**2 * 50
            fr = H * r / (1.0 - (r / R)**2)
            return fr - np.roll(fr, 1)

    Raises RuntimeError if any |r_n| >= R (finite extensibility violated).
    """
    r = np.roll(q, -1) - q  # r_n = q_{n+1} - q_n
    x = r / R
    denom = 1.0 - x**2
    if np.any(denom <= 0.0):
        msg = f"FENE extensibility limit exceeded: max|r/R| = {np.max(np.abs(x)):.4f}"
        raise RuntimeError(msg)
    fr = H * r / denom
    return fr - np.roll(fr, 1)  # F(r_n) - F(r_{n-1})


def _periodic_interp(xi_src: np.ndarray, u_src: np.ndarray, xi_dst: np.ndarray, Lx: float) -> np.ndarray:
    """Linear interpolation of periodic data onto a regular grid."""
    xi_w = xi_src % Lx
    idx = np.argsort(xi_w)
    xs = xi_w[idx]
    us = u_src[idx]
    xs_ext = np.concatenate([xs - Lx, xs, xs + Lx])
    us_ext = np.concatenate([us, us, us])
    return np.interp(xi_dst, xs_ext, us_ext)


# ─── Core simulation ───────────────────────────────────────────────────────────


def simulate_fene_v2(
    u_ic: FENEv2IC,
    *,
    c: float,
    eps: float,
    Lx: float,
    N_chain: int,
    N_kdv: int,
    dt_kdv: float,
    T: float,
    chain_substeps: int,
    save_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate the FENE-v2 chain and return u(xi, tau) on the KdV grid.

    Chain is integrated in lab time using Störmer-Verlet (leapfrog).
    Auto-increases chain_substeps if needed for Verlet stability (c*dt_chain < 1).
    At each saved slow-time step, strains are rescaled and interpolated
    onto the co-moving KdV grid.

    Returns:
        t_slow:      shape (n_frames,), slow time tau = eps^3 * t_lab
        xi_grid:     shape (N_kdv,), co-moving spatial coordinate
        u_series:    shape (n_frames, N_kdv), reconstructed slow field u = r/eps^2
        r_max_seen:  max |r_n / R| observed during simulation
    """
    H = c**2
    R = eps**2 * 50

    # Auto-increase chain_substeps for Verlet stability (c * dt_chain < 1)
    dt_chain_ref = dt_kdv / eps**3
    dt_chain_stable_max = 0.2 / c
    min_substeps = int(np.ceil(dt_chain_ref / dt_chain_stable_max))
    chain_substeps_eff = max(chain_substeps, min_substeps)
    dt_chain = dt_chain_ref / chain_substeps_eff

    n_sites = np.arange(N_chain, dtype=float)
    xi_n0 = eps * n_sites  # slow-frame positions at t_lab = 0

    # KdV output grid
    xi_grid = np.linspace(0.0, Lx, N_kdv, endpoint=False)

    # IC via scaling ansatz: r_n(0) = eps^2 * u0(eps*n), rdot_n(0) = -c*eps^3*u0'(eps*n)
    # Build displacement q by cumulative sum of strain r0 (periodic chain)
    r0 = eps**2 * u_ic.eval(xi_n0)
    rdot0 = -c * eps**3 * u_ic.eval_deriv(xi_n0)
    q = np.zeros(N_chain, dtype=float)
    v = np.zeros(N_chain, dtype=float)
    for n in range(1, N_chain):
        q[n] = q[n - 1] + r0[n - 1]
        v[n] = v[n - 1] + rdot0[n - 1]
    q -= np.mean(q)
    v -= np.mean(v)

    total_kdv_steps = round(T / dt_kdv)
    n_frames = total_kdv_steps // save_every + 1
    u_series = np.zeros((n_frames, N_kdv), dtype=np.float32)
    t_slow = np.zeros(n_frames, dtype=np.float64)

    def reconstruct_u(q: np.ndarray, t_lab: float) -> np.ndarray:
        r = np.roll(q, -1) - q  # strain
        xi_n = eps * n_sites - eps * c * t_lab  # co-moving positions
        return _periodic_interp(xi_n, r / eps**2, xi_grid, Lx)

    # Record t=0
    t_lab = 0.0
    u_series[0] = reconstruct_u(q, t_lab)
    frame_idx = 1
    r_max_seen = 0.0

    for step in range(1, total_kdv_steps + 1):
        for _ in range(chain_substeps_eff):
            acc = _fene_v2_acc(q, H, R)
            v_half = v + 0.5 * dt_chain * acc
            q = q + dt_chain * v_half
            acc_new = _fene_v2_acc(q, H, R)
            v = v_half + 0.5 * dt_chain * acc_new
            t_lab += dt_chain

        r_now = np.roll(q, -1) - q
        r_max_seen = max(r_max_seen, float(np.max(np.abs(r_now))) / R)

        if step % save_every == 0 and frame_idx < n_frames:
            u_series[frame_idx] = reconstruct_u(q, t_lab)
            t_slow[frame_idx] = eps**3 * t_lab
            frame_idx += 1

    return t_slow[:frame_idx], xi_grid, u_series[:frame_idx], r_max_seen


# ─── Data generator ────────────────────────────────────────────────────────────


class FENEv2DataGen:
    r"""Generate dataset of 1D FENE-v2 chain dynamics.

    ======== FENE-v2 chain system ========
    Displacement formulation of a FENE lattice:
      F(r) = H*r / (1 - (r/R)^2),   H = c^2,   R = eps^2 * 50
      r_n = q_{n+1} - q_n,   q_ddot_n = F(r_n) - F(r_{n-1})
    IC via scaling ansatz: r_n(0) = eps^2 * u0(eps*n).
    Output: u(xi, tau) = strain(q)/eps^2 in the co-moving frame.
    """

    info_dict = {"version": 1.0, "preprocess_dag": False, "pde_type_id": 4003}

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.c = args.c
        self.Lx = args.Lx
        self.N_chain = args.N_chain
        self.eps = args.Lx / args.N_chain
        self.N_kdv = args.N_kdv
        self.dt_kdv = args.dt_kdv
        self.T = args.T
        self.chain_substeps = args.chain_substeps
        self.save_every = args.save_every
        self.num_samples = args.num_samples
        self.r_bound = args.r_bound

        self.u_ic = FENEv2IC(args.Lx, args.Nk, args.max_k)

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
        stem = f"custom_v{version:g}_fene_nx{self.N_kdv}_nchain{self.N_chain}"
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

    def run_simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return simulate_fene_v2(
            self.u_ic,
            c=self.c,
            eps=self.eps,
            Lx=self.Lx,
            N_chain=self.N_chain,
            N_kdv=self.N_kdv,
            dt_kdv=self.dt_kdv,
            T=self.T,
            chain_substeps=self.chain_substeps,
            save_every=self.save_every,
        )

    def _accept(self, u_series: np.ndarray, r_max_seen: float, log_fn=None) -> bool:
        if not np.isfinite(u_series).all():
            if callable(log_fn):
                log_fn("rejected: non-finite values")
            return False
        if r_max_seen > self.r_bound:
            if callable(log_fn):
                log_fn(f"rejected: max|r/R| = {r_max_seen:.3f} > {self.r_bound:.3f}")
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
                log_fn(f"trial {trial}, collected {len(self.u_sol_all)}/{self.num_samples}")
            self.reset(rng)
            if callable(log_fn):
                log_fn(f"  IC: {self.u_ic}")
            try:
                t_slow, xi_grid, u_series, r_max_seen = self.run_simulation()
            except RuntimeError as e:
                if callable(log_fn):
                    log_fn(f"  rejected: {e}")
                continue
            if self._accept(u_series, r_max_seen, log_fn):
                self._record(t_slow, xi_grid, u_series)

    # ── Save ───────────────────────────────────────────────────────────────────

    def save_hdf5(self) -> str:
        args = self.args
        os.makedirs(args.h5_file_dir, exist_ok=True)
        stem = self.get_file_stem()
        path = os.path.join(args.h5_file_dir, stem + ".hdf5")

        H = self.c**2
        R = self.eps**2 * 50

        with h5py.File(path, "w") as f:
            f.create_dataset("t_coord", data=self.t_coord)
            f.create_dataset("x_coord", data=self.x_coord)
            # u_sol_all: (num_samples, n_frames, N_kdv, 1) — trailing channel dim for consistency
            f.create_dataset("u_sol_all", data=np.array(self.u_sol_all)[..., None])
            f.create_dataset("coef/u_ic/ks_all", data=np.array(self.coef_ks))
            f.create_dataset("coef/u_ic/amps_all", data=np.array(self.coef_amps))
            f.create_dataset("coef/u_ic/phases_all", data=np.array(self.coef_phases))
            # FENE-v2 physics parameters
            f.create_dataset("pde_info/c", data=self.c)
            f.create_dataset("pde_info/eps", data=self.eps)
            f.create_dataset("pde_info/H", data=H)
            f.create_dataset("pde_info/R", data=R)
            for key, value in self.info_dict.items():
                f.create_dataset(f"pde_info/{key}", data=value)
            for key, value in vars(args).items():
                if isinstance(value, int | float | np.ndarray):
                    f.create_dataset(f"pde_info/args/{key}", data=value)

        return path


# ─── CLI ───────────────────────────────────────────────────────────────────────


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=FENEv2DataGen.__doc__)
    parser.add_argument("--num_samples", "-n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--np_seed", "-r", type=int, default=-1, help="NumPy random seed (-1 = random)")
    parser.add_argument("--r_bound", type=float, default=0.9, help="Reject samples with max|r/R| > r_bound")
    # FENE-v2 parameters
    parser.add_argument("--c", type=float, default=1.0, help="Linear wave speed; spring constant H = c^2")
    # Domain
    parser.add_argument("--Lx", type=float, default=2.0 * np.pi, help="Periodic domain length")
    parser.add_argument("--N_kdv", type=int, default=256, help="Output grid points")
    parser.add_argument("--N_chain", type=int, default=209, help="Number of chain sites (eps = Lx/N_chain)")
    # Time
    parser.add_argument("--dt_kdv", type=float, default=1e-3, help="Slow-time step size")
    parser.add_argument("--T", type=float, default=1.0, help="Total slow time")
    parser.add_argument("--chain_substeps", type=int, default=1, help="Min lab-time substeps (auto-increased for CFL)")
    parser.add_argument("--save_every", type=int, default=10, help="Save u every this many slow-time steps")
    # IC
    parser.add_argument("--Nk", type=int, default=3, help="Number of Fourier modes in IC")
    parser.add_argument("--max_k", type=int, default=6, help="Max Fourier wavenumber in IC")
    # Output
    parser.add_argument("--h5_file_dir", type=str, default="data/fene")
    return parser.parse_args()


def _create_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("fene_v2")
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

    gen = FENEv2DataGen(args)

    os.makedirs("log", exist_ok=True)
    logger = _create_logger(os.path.join("log", gen.get_file_stem() + ".log"))
    logger.info("target file: %s.hdf5", gen.get_file_stem())

    eps = gen.eps
    H = gen.c**2
    R = eps**2 * 50
    dt_chain_ref = args.dt_kdv / eps**3
    dt_chain_stable_max = 0.2 / gen.c
    min_substeps = int(np.ceil(dt_chain_ref / dt_chain_stable_max))
    chain_substeps_eff = max(args.chain_substeps, min_substeps)
    dt_chain = dt_chain_ref / chain_substeps_eff

    logger.info("eps = %.5f  (N_chain=%d, Lx=%.4f)", eps, args.N_chain, args.Lx)
    logger.info("c = %.4f,  H = c^2 = %.4f,  R = eps^2*50 = %.6f", gen.c, H, R)
    logger.info("chain_substeps = %d (auto from %d)", chain_substeps_eff, args.chain_substeps)
    logger.info("dt_chain = %.3e  (CFL = c*dt_chain = %.3f, should be < 1)", dt_chain, gen.c * dt_chain)

    rng = np.random.default_rng(None if args.np_seed == -1 else args.np_seed)
    gen.gen_data_all(rng, log_fn=logger.info)
    path = gen.save_hdf5()
    logger.info("file saved: %s", path)
