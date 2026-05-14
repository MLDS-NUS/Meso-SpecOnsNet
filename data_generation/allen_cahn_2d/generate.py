r"""Generate dataset of 2D Allen-Cahn equation with periodic boundary conditions.

Allen-Cahn (2D):
    u_t = Δu + (1/ε²)(u - u³),
    (t, x, y) in [0, T] x [0, Lx] x [0, Ly], periodic in x and y.

Uses Dedalus-v3 with two RealFourier bases (periodic in both directions).
The linear Laplacian is treated implicitly (SBDF2) and the cubic source explicitly,
so the timestep restriction comes only from the nonlinear term: dt ≲ ε² / max|3u² − 1|.
With ε = 0.1 and |u| ≤ 1, dt ≲ ~0.005; the default dt = 4e-5 mirrors the 1D variant.

ref:
[1] https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation
[2] Bartels, Sören (2015). Numerical Methods for Nonlinear Partial Differential Equations. Springer.
"""

import argparse
import logging
import os
import sys
import time

import dedalus.public as d3
import h5py
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ─── Default domain / resolution ───────────────────────────────────────────────

LX_DEFAULT = 2 * np.pi
LY_DEFAULT = 2 * np.pi
NX_DEFAULT = 128
NY_DEFAULT = 128
DEALIAS = 2

# Module-level Dedalus distributor (one per process, shared across runs).
coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=np.float64)


# ─── Initial condition ─────────────────────────────────────────────────────────


class InitialCondition2D:
    """Random smooth periodic initial condition u0(x, y).

    Built as a sum of Nk plane waves with random integer wavenumbers (kx, ky)
    drawn without replacement from {-max_k, …, max_k}^2 \\ {(0,0)}, random
    amplitudes in [0, 1] and random phases in [0, 2π). Finally rescaled so that
    the field is in [-1, 1] (the bistable wells of the Allen-Cahn potential).
    """

    def __init__(self, Lx: float, Ly: float, Nk: int, max_k: int) -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.Nk = Nk
        self.max_k = max_k

    def reset(self, rng: np.random.Generator) -> None:
        """Draw a new random set of (kx, ky, amp, phase)."""
        all_modes = [
            (kx, ky)
            for kx in range(-self.max_k, self.max_k + 1)
            for ky in range(-self.max_k, self.max_k + 1)
            if (kx, ky) != (0, 0)
        ]
        idx = rng.choice(len(all_modes), size=self.Nk, replace=False)
        self.kxs = np.array([all_modes[i][0] for i in idx], dtype=np.int64)
        self.kys = np.array([all_modes[i][1] for i in idx], dtype=np.int64)
        self.amps = rng.uniform(0.0, 1.0, size=self.Nk)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.Nk)

    def eval(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """Evaluate u0 on a 2D grid.

        x_grid and y_grid may either be 1D arrays (shape (Nx,) and (Ny,)) or
        already broadcastable arrays as returned by Dedalus' `dist.local_grid`.
        Returns a 2D array of shape (Nx, Ny), normalised to [-1, 1].
        """
        x_arr = np.asarray(x_grid, dtype=float)
        y_arr = np.asarray(y_grid, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[None, :]

        out_shape = np.broadcast_shapes(x_arr.shape, y_arr.shape)
        field = np.zeros(out_shape, dtype=float)
        for kx, ky, a, phi in zip(self.kxs, self.kys, self.amps, self.phases):
            theta = (2.0 * np.pi * kx / self.Lx) * x_arr + (2.0 * np.pi * ky / self.Ly) * y_arr + phi
            field = field + a * np.sin(theta)

        fmin = float(field.min())
        fmax = float(field.max())
        if fmax - fmin > 1e-12:
            field = 2.0 * (field - fmin) / (fmax - fmin) - 1.0
        return field

    def get_data_dict(self, prefix: str) -> dict:
        return {
            prefix + "/kxs": self.kxs,
            prefix + "/kys": self.kys,
            prefix + "/amps": self.amps,
            prefix + "/phases": self.phases,
        }

    def __str__(self) -> str:
        if not hasattr(self, "kxs"):
            return "InitialCondition2D(uninitialized)"
        modes = list(zip(self.kxs.tolist(), self.kys.tolist()))
        return f"InitialCondition2D(modes={modes}, amps={self.amps.round(3)})"


# ─── Core simulation ───────────────────────────────────────────────────────────


def simulate_allen_cahn_2d(
    u_ic: InitialCondition2D,
    *,
    eps: float,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    T: float,
    dt: float,
    save_every: int,
    timestepper=d3.SBDF2,
    dealias: int = DEALIAS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate 2D Allen-Cahn on a regular periodic grid.

    Returns
    -------
    t_coord : (n_frames,) float64
    x_coord : (Nx,) float64
    y_coord : (Ny,) float64
    u_series: (n_frames, Nx, Ny) float32
    """
    xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias)
    x_grid = dist.local_grid(xbasis)
    y_grid = dist.local_grid(ybasis)
    x_coord = np.asarray(x_grid).ravel().copy()
    y_coord = np.asarray(y_grid).ravel().copy()

    u = dist.Field(name="u", bases=(xbasis, ybasis))
    u["g"] = u_ic.eval(x_grid, y_grid)

    nonlin = (u - u**3) / eps**2
    problem = d3.IVP([u])
    problem.add_equation([d3.dt(u) - d3.lap(u), nonlin])

    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = T

    t_list: list[float] = []
    u_list: list[np.ndarray] = []

    def record() -> None:
        u.change_scales(1)
        t_list.append(float(solver.sim_time))
        u_list.append(np.copy(u["g"]).astype(np.float32))

    record()
    step = 0
    while solver.proceed:
        solver.step(dt)
        step += 1
        if step % save_every == 0:
            record()

    return (
        np.asarray(t_list, dtype=np.float64),
        x_coord,
        y_coord,
        np.stack(u_list, axis=0),
    )


# ─── Data generator ────────────────────────────────────────────────────────────


class AllenCahn2dDataGen:
    r"""Generate a dataset of 2D Allen-Cahn solutions on a periodic torus.

    ======== Allen-Cahn (2D) ========
        u_t - Δu - (1/ε²)(u - u³) = 0,  (x, y) periodic.
    Initial condition: random smooth periodic field, normalised to [-1, 1].
    Stored field shape: u_sol_all[i] has shape (n_frames, Nx, Ny, 1) — trailing
    channel dim for consistency with the 1D datasets.
    """

    info_dict = {"version": 1.0, "preprocess_dag": False, "pde_type_id": 2001}

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.eps = args.eps
        self.Lx = args.Lx
        self.Ly = args.Ly
        self.Nx = args.Nx
        self.Ny = args.Ny
        self.T = args.T
        self.dt = args.dt
        self.save_every = args.save_every
        self.num_samples = args.num_samples
        self.u_bound = args.u_bound

        self.u_ic = InitialCondition2D(args.Lx, args.Ly, args.Nk, args.max_k)

        self.t_coord: np.ndarray | None = None
        self.x_coord: np.ndarray | None = None
        self.y_coord: np.ndarray | None = None
        self.u_sol_all: list[np.ndarray] = []
        self.coef_kxs: list[np.ndarray] = []
        self.coef_kys: list[np.ndarray] = []
        self.coef_amps: list[np.ndarray] = []
        self.coef_phases: list[np.ndarray] = []

    # ── Filename ───────────────────────────────────────────────────────────────

    def get_file_stem(self) -> str:
        args = self.args
        version = self.info_dict["version"]
        stem = f"custom_v{version:g}_allen_cahn_2d_nx{self.Nx}_ny{self.Ny}"
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

    def run_simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return simulate_allen_cahn_2d(
            self.u_ic,
            eps=self.eps,
            Lx=self.Lx,
            Ly=self.Ly,
            Nx=self.Nx,
            Ny=self.Ny,
            T=self.T,
            dt=self.dt,
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
                log_fn(f"rejected: |u|_max = {u_max:.3f} > {self.u_bound:.3f}")
            return False
        return True

    def _record(
        self,
        t_coord: np.ndarray,
        x_coord: np.ndarray,
        y_coord: np.ndarray,
        u_series: np.ndarray,
    ) -> None:
        if self.t_coord is None:
            self.t_coord = t_coord
            self.x_coord = x_coord
            self.y_coord = y_coord
        self.u_sol_all.append(u_series.astype(np.float32))
        coef = self.u_ic.get_data_dict("u_ic")
        self.coef_kxs.append(coef["u_ic/kxs"])
        self.coef_kys.append(coef["u_ic/kys"])
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
                t_coord, x_coord, y_coord, u_series = self.run_simulation()
            except RuntimeError as e:
                if callable(log_fn):
                    log_fn(f"  simulation failed: {e}")
                continue
            if self._accept(u_series, log_fn):
                self._record(t_coord, x_coord, y_coord, u_series)

    # ── Save ───────────────────────────────────────────────────────────────────

    def save_hdf5(self) -> str:
        args = self.args
        os.makedirs(args.h5_file_dir, exist_ok=True)
        stem = self.get_file_stem()
        path = os.path.join(args.h5_file_dir, stem + ".hdf5")

        with h5py.File(path, "w") as f:
            f.create_dataset("t_coord", data=self.t_coord)
            f.create_dataset("x_coord", data=self.x_coord)
            f.create_dataset("y_coord", data=self.y_coord)
            # u_sol_all: (num_samples, n_frames, Nx, Ny, 1) — trailing channel dim for consistency.
            f.create_dataset("u_sol_all", data=np.array(self.u_sol_all)[..., None])
            f.create_dataset("coef/u_ic/kxs_all", data=np.array(self.coef_kxs))
            f.create_dataset("coef/u_ic/kys_all", data=np.array(self.coef_kys))
            f.create_dataset("coef/u_ic/amps_all", data=np.array(self.coef_amps))
            f.create_dataset("coef/u_ic/phases_all", data=np.array(self.coef_phases))
            f.create_dataset("pde_info/eps", data=self.eps)
            for key, value in self.info_dict.items():
                f.create_dataset(f"pde_info/{key}", data=value)
            for key, value in vars(args).items():
                if isinstance(value, int | float | np.ndarray):
                    f.create_dataset(f"pde_info/args/{key}", data=value)

        return path


# ─── CLI ───────────────────────────────────────────────────────────────────────


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=AllenCahn2dDataGen.__doc__)
    parser.add_argument("--num_samples", "-n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--np_seed", "-r", type=int, default=-1, help="NumPy random seed (-1 = random)")
    parser.add_argument("--u_bound", type=float, default=10.0, help="Reject samples with max|u| > u_bound")
    # Allen-Cahn parameter
    parser.add_argument("--eps", type=float, default=0.1, help="Phase-field thickness ε")
    # Domain
    parser.add_argument("--Lx", type=float, default=LX_DEFAULT, help="Periodic domain length in x")
    parser.add_argument("--Ly", type=float, default=LY_DEFAULT, help="Periodic domain length in y")
    parser.add_argument("--Nx", type=int, default=NX_DEFAULT, help="Grid points in x")
    parser.add_argument("--Ny", type=int, default=NY_DEFAULT, help="Grid points in y")
    # Time
    parser.add_argument("--T", type=float, default=0.1, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=4e-5, help="Time step")
    parser.add_argument("--save_every", type=int, default=25, help="Save every this many time steps")
    # IC
    parser.add_argument("--Nk", type=int, default=5, help="Number of (kx, ky) modes in IC")
    parser.add_argument("--max_k", type=int, default=5, help="Max |kx|, |ky| in IC")
    # Output
    parser.add_argument("--h5_file_dir", type=str, default="data/allen_cahn_2d")
    return parser.parse_args()


def _create_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("allen_cahn_2d")
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

    gen = AllenCahn2dDataGen(args)

    os.makedirs("log", exist_ok=True)
    logger = _create_logger(os.path.join("log", gen.get_file_stem() + ".log"))
    logger.info("target file: %s.hdf5", gen.get_file_stem())
    logger.info("eps=%.4f, Lx=%.4f, Ly=%.4f, Nx=%d, Ny=%d", args.eps, args.Lx, args.Ly, args.Nx, args.Ny)
    logger.info("T=%.4f, dt=%.5g, save_every=%d", args.T, args.dt, args.save_every)

    rng = np.random.default_rng(None if args.np_seed == -1 else args.np_seed)
    gen.gen_data_all(rng, log_fn=logger.info)
    path = gen.save_hdf5()
    logger.info("file saved: %s", path)
