r"""Generate dataset of 1D KdV equation with periodic boundary condition.

KdV: u_t + 6 u u_x + u_xxx = 0,  (t,x) in [0, T] x [X_L, X_R], periodic in x.
Uses Dedalus-v3 with RealFourier basis (periodic).
"""

import argparse
import sys
from functools import partial

import dedalus.public as d3
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data_generation.common import InitialCondition, PDEDataGenBase, PDETermBase, gen_data  # noqa: E402

# Domain / resolution (match your project conventions as needed)
Lx = 2 * np.pi
Nx = 256
DEALIAS = 2

# Dedalus setup
d3_xcoord = d3.Coordinate("x")
dist = d3.Distributor(d3_xcoord, dtype=np.float64)


def d3_dx(operand):
    return d3.Differentiate(operand, d3_xcoord)


class KdVPDE(PDEDataGenBase):
    r"""
    Generate dataset of 1D time-dependent KdV solutions with Dedalus-v3.
    ======== KdV ========
    u_t + 6 u u_x + u_xxx = 0, x periodic.
    """

    info_dict = {"version": 1.0, "preprocess_dag": False, "pde_type_id": 1001}

    # You can override these if you want different time horizon / sampling rate
    STOP_SIM_TIME = 0.1
    T_SAVE_STEPS = 25
    TIMESTEP = 1e-3 / T_SAVE_STEPS
    TIMESTEPPER = d3.SBDF2  # implicit/IMEX style; works well for high derivatives

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        # Periodic Fourier basis
        self.xbasis = d3.RealFourier(d3_xcoord, size=Nx, bounds=(0, Lx), dealias=DEALIAS)
        self.x_coord = dist.local_grid(self.xbasis)

        # PDE terms: just the initial condition (random periodic field)
        self.u_ic = InitialCondition(
            self.x_coord,
            Nk=args.Nk,
            max_k=args.max_k,
        )

    @property
    def _term_obj_dict(self) -> dict[str, PDETermBase]:
        # Recorded into HDF5 under coef/u_ic/*
        return {"u_ic": self.u_ic}

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return f"kdv_circ_nx{Nx}"

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument(
            "--max_k",
            type=int,
            default=6,
            help="Max Fourier modes available for IC generation",
        )
        parser.add_argument(
            "--Nk",
            type=int,
            default=3,
            help="Number of Fourier modes included for IC generation",
        )
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.u_sol = None
        self.u_ic.reset(rng)
        # self.u_ic.field *= 0.1  # smaller initial condition

    def _get_dedalus_problem(self) -> tuple:
        # Field
        u = dist.Field(name="u", bases=self.xbasis)
        u_list = [u]

        # Initial condition (sets u['g'])
        u = self.u_ic.gen_dedalus_ops(u)

        dx = partial(d3.Differentiate, coord=d3.Coordinate("x"))

        # Build KdV: u_t + u_xxx + 6 u u_x = 0
        u_xxx = dx(dx(dx(u)))
        nonlin = 3 * dx(u**2)  # since 6 u u_x = 3 (u^2)_x

        problem = d3.IVP([u])
        problem.add_equation([d3.dt(u) + u_xxx, -nonlin])
        return u_list, problem


if __name__ == "__main__":
    args = KdVPDE.get_cli_args()

    if "--h5_file_dir" not in sys.argv:
        args.h5_file_dir = "data/kdv"

    pde_data_gen = KdVPDE(args)
    gen_data(args, pde_data_gen)
