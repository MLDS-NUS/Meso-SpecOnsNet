# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common utilities for PDE data generation with periodic BC and random field ICs.
ref: https://gitee.com/mindspore/mindscience/blob/legacy-master/MindFlow/applications/pdeformer1d/data_generation/README.md
"""

import argparse
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import dedalus.public as d3
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class PDETermBase(ABC):
    r"""Abstract base class for PDE terms (e.g., initial conditions)."""

    def __str__(self) -> str:
        data_dict = self.get_data_dict(" ")
        str_list = [f"{key}: {data}" for key, data in data_dict.items()]
        return " ".join(str_list)

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        r"""Reset the random parameters in the current term.

        Args:
            rng: Random number generator instance.
        """

    @abstractmethod
    def gen_dedalus_ops(self):
        r"""Generate the operator for the Dedalus solver."""
        return 0

    @abstractmethod
    def get_data_dict(self, prefix: str) -> dict[str, int | float | NDArray]:
        r"""Return a dictionary containing the current parameters in this PDE term."""
        return {}

    def prepare_plot(self, title: str = "") -> None:  # noqa: B027
        r"""Create a matplotlib figure showing the current parameters.

        Use `plt.show()` afterwards to see the results.
        """


class InitialCondition(PDETermBase):
    r"""Random field initial condition generator for periodic PDEs.

    Generates smooth random fields as superpositions of sine waves with
    random amplitudes, frequencies, and phases.
    """

    FIELD_COEF = 2
    coef_type: int
    field: NDArray[np.float64]
    x_coord: NDArray[np.float64]
    max_k: int
    Nk: int

    def __init__(self, x_coord: NDArray[np.float64], Nk: int, max_k: int) -> None:
        self.x_coord = x_coord
        self.Nk = Nk
        self.max_k = max_k
        if not np.all(np.diff(x_coord) > 0):
            raise ValueError("'x_coord' should be strictly increasing.")

    def __str__(self) -> str:
        data = self.field.flat
        return f"field [{data[0]:.4g} ... {data[-1]:.4g}], shape {self.field.shape}"

    @staticmethod
    def _rand_frequencies(
        rng: np.random.Generator,
        x_len: int,
        bs: int,
        max_k: int,
        Nk: int,
    ) -> NDArray[np.float64]:
        r"""Generate random frequencies for the periodic random field.

        Args:
            rng: Random number generator.
            x_len: Length of spatial domain.
            numbers: Number of random fields to generate.
            k_tot: Total number of frequency modes available.
            num_choice_k: Number of modes to randomly select for each field.

        Returns:
            Array of angular frequencies with shape [numbers, k_tot].
        """
        if x_len <= 0:
            raise ValueError(f"'x_len' should be positive, but got {x_len}.")

        # Select random modes from 1 to k_tot
        # selected = rng.integers(size=[bs, Nk], low=0, high=max_k)
        selected = np.stack([rng.choice(max_k, size=(Nk,), replace=False) for _ in range(bs)], axis=0)

        selected = np.eye(max_k)[selected].sum(axis=1)
        # (max_k, max_k) -> (bs, Nk, max_k) -> (bs, max_k)
        angular_freq = np.pi * 2.0 * np.arange(1, max_k + 1) * selected / x_len

        # print(f"{selected=}")

        return angular_freq

    @classmethod
    def _random_field(
        cls,
        x_coord: NDArray[np.float64],
        rng: np.random.Generator,
        max_k: int,
        Nk: int,
        bs: int = 1,
    ) -> NDArray[np.float64]:
        r"""Generate random fields as superpositions of sine waves.

        Args:
            x_coord: Cell center coordinates.
            rng: Random number generator.
            k_tot: Total number of frequency modes available.
            num_choice_k: Number of modes to randomly select.
            numbers: Number of random fields to generate.

        Returns:
            Random field(s) with shape [numbers, n_x_grid].

        Note:
            Code adapted from PDEBench.
        """
        # Generate angular frequencies
        x_len = x_coord[-1] - x_coord[0]
        angular_freq = cls._rand_frequencies(rng, x_len, bs, max_k, Nk)

        # Sum of sine waves with random amplitudes and phases
        amp = rng.uniform(size=[bs, max_k, 1])
        phs = 2.0 * np.pi * rng.uniform(size=[bs, max_k, 1])
        u_ = amp * np.sin(angular_freq[:, :, None] * x_coord[None, None, :] + phs)
        # [numbers, k_tot, n_x_grid] -> [numbers, n_x_grid]
        u_ = np.sum(u_, axis=1)

        # normalize to std=1
        u_ = u_ / np.std(u_, axis=-1, keepdims=True)

        return u_

    def reset(self, rng: np.random.Generator) -> None:
        """Generate a new random field initial condition."""
        self.coef_type = self.FIELD_COEF
        field = self._random_field(self.x_coord, rng, self.max_k, self.Nk)
        (self.field,) = field  # [1, n_x_grid] -> [n_x_grid]

    def gen_dedalus_ops(self, field_op):
        """Set the initial condition in the Dedalus field operator."""
        field_op["g"] = self.field
        return field_op

    def get_data_dict(self, prefix: str) -> dict[str, int | NDArray[np.float64]]:
        """Return the field data for saving to HDF5."""
        return {prefix + "/coef_type": self.coef_type, prefix + "/field": self.field}

    def prepare_plot(self, title: str = "") -> None:
        """Plot the initial condition field."""
        plt.figure()
        plt.plot(self.x_coord, self.field)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("u(0, x)")


class PDEDataGenBase(ABC):
    r"""Base class for generating PDE solution datasets with Dedalus-v3.

    This class handles:
    - PDE solution generation with Dedalus
    - Data collection and validation
    - HDF5 file saving
    - Logging and visualization
    """

    info_dict = {"version": 4.2, "preprocess_dag": False, "pde_type_id": 0}
    n_vars: int = 1  # number of components in the PDE

    # Dedalus Parameters (can be overridden in subclasses)
    STOP_SIM_TIME = 1.0
    TIMESTEPPER = d3.SBDF2
    T_SAVE_STEPS = 25
    TIMESTEP = 1e-2 / T_SAVE_STEPS

    # Solution data
    t_coord: NDArray[np.float64]
    x_coord: NDArray[np.float64]
    u_sol: NDArray[np.float64] | None

    def __init__(self, args: argparse.Namespace) -> None:
        self.u_bound = args.u_bound
        self.num_pde = args.num_pde
        self.print_coef_level = args.print_coef_level
        self.plot_u = args.plot_u
        self.u_sol_all = []
        self.coef_all_dict = {}

    @property
    def current_coef_dict(self) -> dict[str, int | float | NDArray]:
        r"""A dictionary containing the current PDE coefficients."""
        coef_dict = {}
        for prefix, term_obj in self._term_obj_dict.items():
            coef_dict.update(term_obj.get_data_dict(prefix))
        return coef_dict

    @property
    def current_coef_str_dict(self) -> dict[str, str]:
        r"""A dictionary containing the string representation of the current PDE coefficients."""
        str_dict = {prefix: str(term_obj) for prefix, term_obj in self._term_obj_dict.items()}
        return str_dict

    @property
    def n_data(self) -> int:
        r"""Current number of generated PDE solution samples."""
        return len(self.u_sol_all)

    @property
    @abstractmethod
    def _term_obj_dict(self) -> dict[str, PDETermBase]:
        r"""A dictionary containing the PDE terms, as instances of `PDETermBase`."""
        return {}

    @staticmethod
    @abstractmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        """Get the prefix for the HDF5 filename (e.g., 'kdv_circ_nx256')."""
        return "base"

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        """Get the argument parser for this PDE type."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        return parser

    @classmethod
    def get_cli_args(cls) -> argparse.Namespace:
        r"""Parse the command-line arguments for this PDE."""
        parser = cls._get_cli_args_parser()
        parser.add_argument(
            "--num_pde",
            "-n",
            type=int,
            default=10000,
            help="number of PDEs to generate",
        )
        parser.add_argument("--np_seed", "-r", type=int, default=-1, help="NumPy random seed")
        parser.add_argument(
            "--u_bound",
            type=float,
            default=10,
            help="accept only solutions with |u| <= u_bound",
        )
        parser.add_argument(
            "--print_coef_level",
            type=int,
            default=0,
            choices=[0, 1, 2, 3],
            help=r"""
                            print coefficients of each generated PDE.
                            0: do not print. 1: print human-readable results.
                            2: print raw data. 3: print both human-readable
                            results and raw data. """,
        )
        parser.add_argument(
            "--plot_u",
            action="store_true",
            help="show image(s) of each accepted solution",
        )
        parser.add_argument("--h5_file_dir", type=str, default="results")
        args = parser.parse_args()
        return args

    @classmethod
    def get_hdf5_file_name(cls, args: argparse.Namespace) -> str:
        r"""Get the name of the target HDF5 file where the generated data is stored."""
        if "version" not in cls.info_dict:
            raise KeyError("subclass of PDEDataGenBase should provide 'info_dict' with 'version' key.")
        version = cls.info_dict["version"]
        fname = f"custom_v{version:g}_"
        fname += cls._get_hdf5_file_prefix(args)
        if args.num_pde != 10000:
            fname += f"_num{args.num_pde}"
        if args.np_seed == -1:
            fname += time.strftime("_%Y-%m-%d-%H-%M-%S")
        else:
            fname += f"_seed{args.np_seed}"
        return fname

    @abstractmethod
    def reset_pde(self, rng: np.random.Generator) -> None:
        r"""Randomly reset the terms in the current PDE (e.g., initial condition)."""
        self.u_sol = None

    def print_current_coef(self, print_fn: None | Callable = print) -> None:
        r"""Print the current coefficients/parameters of the PDE."""
        if not (self.print_coef_level > 0 and callable(print_fn)):
            return
        if self.print_coef_level in [1, 3]:
            print_str = "coefs: "
            for prefix, coef_str in self.current_coef_str_dict.items():
                print_str += f"\n  {prefix}: {coef_str}"
            print_fn(print_str)
        if self.print_coef_level in [2, 3]:
            print_str = "raw coefs: "
            for key, data in self.current_coef_dict.items():
                if np.size(data) < 20:
                    print_str += f"\n  {key}: {data}"
            print_fn(print_str)

    def gen_solution(self) -> None:
        r"""Generate the PDE solution corresponding to the current parameters."""
        u_list, problem = self._get_dedalus_problem()

        # Solver
        solver = problem.build_solver(self.TIMESTEPPER)
        solver.stop_sim_time = self.STOP_SIM_TIME

        # Main loop
        u_sol, t_coord = [], []

        def record():
            for ui_op in u_list:
                ui_op.change_scales(1)
            u_current = [np.copy(ui_op["g"]) for ui_op in u_list]
            u_sol.append(np.stack(u_current, axis=-1))
            t_coord.append(solver.sim_time)

        record()
        while solver.proceed:
            solver.step(self.TIMESTEP)
            if solver.iteration % self.T_SAVE_STEPS == 0:
                record()

        self.t_coord = np.array(t_coord)
        self.u_sol = np.array(u_sol, dtype=np.float32)

    def plot(self, save_pdf_prefix: str | None = None, plot_coef: bool = True) -> None:
        r"""Plot the current PDE solution as well as the coefficients (optional)."""
        # plot coefficients
        if plot_coef:
            for prefix, term_obj in self._term_obj_dict.items():
                term_obj.prepare_plot(prefix)

        # plot current solution
        for i in range(self.n_vars):
            self._prepare_plot_2d(self.u_sol[:, :, i], f"u_sol component {i}")
            if save_pdf_prefix is not None:
                plt.savefig(f"{save_pdf_prefix}_{i}.pdf")

        plt.show()

    def gen_data_all(self, rng: np.random.Generator, print_fn: Callable | None = None) -> None:
        r"""Generate the PDE solution data, until the target number of samples is reached."""
        i_pde = 0
        while self.n_data < self.num_pde:
            i_pde += 1
            if callable(print_fn):
                print_fn(f"trial No. {i_pde}, PDE generated {self.n_data}/{self.num_pde}")
            self.reset_pde(rng)
            self.print_current_coef(print_fn)
            self.gen_solution()
            if self._accept_u_sol(print_fn):
                if self.plot_u:
                    self.plot()
                self._record_solution(self.u_sol)

    def save_hdf5(self, args: argparse.Namespace) -> None:
        r"""Save the generated data to the target HDF5 file."""
        os.makedirs(args.h5_file_dir, exist_ok=True)
        fname = self.get_hdf5_file_name(args)
        h5_filepath = os.path.join(args.h5_file_dir, fname + ".hdf5")

        with h5py.File(h5_filepath, "w") as h5_file:
            h5_file.create_dataset("t_coord", data=self.t_coord)
            h5_file.create_dataset("x_coord", data=self.x_coord)
            h5_file.create_dataset("u_sol_all", data=np.array(self.u_sol_all))
            for key, data_list in self.coef_all_dict.items():
                h5_file.create_dataset("coef/" + key + "_all", data=np.array(data_list))
            for key, value in self.info_dict.items():
                h5_file.create_dataset("pde_info/" + key, data=value)
            for key, value in vars(args).items():
                if isinstance(value, int | float | np.ndarray):
                    h5_file.create_dataset("pde_info/args/" + key, data=value)

    @abstractmethod
    def _get_dedalus_problem(self) -> tuple:
        """Build and return the Dedalus problem for this PDE.

        Returns:
            Tuple of (field_list, problem) where field_list contains the solution
            fields and problem is the Dedalus IVP problem.
        """

    def _accept_u_sol(self, print_fn: Callable | None = None) -> bool:
        """Check if the current solution should be accepted (finite and bounded)."""
        if not np.isfinite(self.u_sol).all():
            return False
        u_max = np.max(np.abs(self.u_sol))
        if u_max > self.u_bound:
            if callable(print_fn):
                print_fn(f"rejected: u_max {u_max:.2f}")
            return False
        return True

    def _prepare_plot_2d(
        self,
        field_2d: NDArray[np.float64],
        title: str = "",
    ) -> None:
        r"""Prepare a 2D plot of the field."""
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(
            self.x_coord,
            self.t_coord,
            field_2d,
            shading="gouraud",
            cmap=plt.get_cmap("jet"),
            rasterized=True,
        )
        plt.ylim(0, self.STOP_SIM_TIME)
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    def _record_solution(self, u_sol: NDArray[np.float64]) -> None:
        """Record the current solution and its parameters."""
        coef_dict = self.current_coef_dict
        if self.n_data == 0:
            for key in coef_dict:
                self.coef_all_dict[key] = []
        for key, value in coef_dict.items():
            self.coef_all_dict[key].append(value)
        self.u_sol_all.append(u_sol)


def create_logger(path: str = "./log.log") -> logging.RootLogger:
    r"""Create and configure a logger that saves to a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = path
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def gen_data(args: argparse.Namespace, pde_data_gen: PDEDataGenBase) -> None:
    r"""Main data generation process.

    Args:
        args: Command-line arguments.
        pde_data_gen: PDE data generator instance.
    """
    # random number generator
    if args.np_seed == -1:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.np_seed)
        np.random.seed(args.np_seed)  # for spm1d.rft1d

    # logger
    os.makedirs("log", exist_ok=True)
    fname = pde_data_gen.get_hdf5_file_name(args)
    logger = create_logger(os.path.join("log", fname + ".log"))
    logger.info("target file: %s.hdf5", fname)

    # generate data and save
    pde_data_gen.gen_data_all(rng, print_fn=logger.info)
    pde_data_gen.save_hdf5(args)
    logger.info("file saved: %s.hdf5", fname)
