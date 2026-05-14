"""
Dataset + DataModule for Dedalus-generated trajectory HDF5 files.

Expected HDF5 structure (from common.py):
  - u_sol_all: (..., nt, *spatial, n_vars)  # leading dims may be (num_pde,) or (num_pde, num_sample_per_pde), etc.
  - t_coord: (nt,)
  - x_coord: (nx,)
  - y_coord: (ny,)            # 2D datasets only
"""

from __future__ import annotations

import bisect
import glob
import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import tensordict as td
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, random_split


def _expand_files(files_or_glob: str | Sequence[str]) -> list[str]:
    if isinstance(files_or_glob, str):
        # allow (e.g.) "data/kdv/*.hdf5"
        files = sorted(glob.glob(files_or_glob))
        if not files and os.path.isfile(files_or_glob):
            files = [files_or_glob]
    else:
        files = list(files_or_glob)

    if not files:
        raise FileNotFoundError(f"No HDF5 files found for: {files_or_glob}")

    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing HDF5 file: {f}")
    return files


class DedalusDynamicsDatabase(Dataset):
    """
    Reads trajectories from one or more Dedalus-generated HDF5 files.

    Each item is a single trajectory tensor with shape: (nt, nx, n_vars)
    """

    def __init__(
        self,
        h5_files: str | Sequence[str],
        dtype: torch.dtype = torch.float32,
    ):
        self.files = _expand_files(h5_files)
        self.dtype = dtype

        #!
        self.dataset_key = dataset_key = "u_sol_all"

        # Build an index mapping: global_idx -> (file_idx, local_flat_idx)
        self._counts: list[int] = []
        self._cum_counts: list[int] = [0]
        self._shapes: list[tuple[int, ...]] = []  # shape of u_sol_all for each file

        # Optional: expose coords (from the first file)
        self.t_coord: np.ndarray | None = None
        self.x_coord: np.ndarray | None = None
        self.y_coord: np.ndarray | None = None

        for path in self.files:
            with h5py.File(path, "r") as f:
                if dataset_key not in f:
                    raise KeyError(f"'{dataset_key}' not found in {path}")

                shape = tuple(f[dataset_key].shape)
                if len(shape) < 3:
                    raise ValueError(f"{path}: {dataset_key} must have >=3 dims, got {shape}")

                # Trailing dims are (nt, *spatial, n_vars).
                n_traj = shape[0]

                self._shapes.append(shape)
                self._counts.append(n_traj)
                self._cum_counts.append(self._cum_counts[-1] + n_traj)

                if self.t_coord is None and "t_coord" in f:
                    self.t_coord = np.array(f["t_coord"])
                if self.x_coord is None and "x_coord" in f:
                    self.x_coord = np.array(f["x_coord"])
                if self.y_coord is None and "y_coord" in f:
                    self.y_coord = np.array(f["y_coord"])

        self.n_trajectories = self._cum_counts[-1]

        # HDF5 handles are process-local; open lazily per worker/process
        self._handles: list[h5py.File | None] = [None] * len(self.files)

    def __len__(self) -> int:
        return self.n_trajectories

    def _get_handle(self, file_idx: int) -> h5py.File:
        h = self._handles[file_idx]
        if h is None:
            # Each DataLoader worker gets its own dataset copy -> safe to open here
            self._handles[file_idx] = h5py.File(self.files[file_idx], "r")
            h = self._handles[file_idx]
        return h

    def __getitem__(self, index: int) -> td.TensorDict:
        if index < 0 or index >= self.n_trajectories:
            raise IndexError(index)

        # find which file this index belongs to
        file_idx = bisect.bisect_right(self._cum_counts, index) - 1
        local = index - self._cum_counts[file_idx]

        h = self._get_handle(file_idx)
        arr = h[self.dataset_key][local]  # -> (nt, *spatial, n_vars)

        arr = torch.from_numpy(arr).to(dtype=self.dtype)
        # Move trailing channel dim into position 1: (nt, *spatial, nvar) -> (nt, nvar, *spatial).
        arr = arr.movedim(-1, 1)

        data = td.TensorDict(
            {"seq": arr},
            batch_size=arr.shape[0],
        )

        return data

    def __del__(self):
        # Close any opened file handles
        for h in getattr(self, "_handles", []):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass


class DedalusDynamicsDataset(Dataset):
    """Yields chunks of consecutive timesteps from a trajectory."""

    def __init__(self, database: Dataset, chunk_length: int):
        super().__init__()
        self.database = database
        self.chunk_length = chunk_length

    def __len__(self) -> int:
        return len(self.database)

    def __getitem__(self, index: int) -> torch.Tensor:
        traj = self.database[index]  # (nt, ...)

        if self.chunk_length == -1:
            return traj

        if 0 < self.chunk_length <= traj.shape[0]:
            s = torch.randint(0, traj.shape[0] - self.chunk_length + 1, (1,)).item()
            return traj[s : s + self.chunk_length]

        raise ValueError(f"Invalid chunk_length {self.chunk_length} for trajectory length {traj.shape[0]}")


class DedalusDynamicsDataModule(LightningDataModule):
    """
    Lightning DataModule for pre-generated HDF5 trajectories.

    Provide either:
      - a glob:   data_files="data/kdv/*.hdf5"
      - a list:   data_files=["data/kdv/a.hdf5", "data/kdv/b.hdf5"]
    """

    def __init__(
        self,
        dyn_type: str,
        data_files: str | Sequence[str],
        split_ratio: tuple[float, float, float],
        batch_size: int,
        chunk_length: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()

        if abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError("Train/val/test split ratios must sum to 1.0")

        self.dyn_type = dyn_type
        self.data_files = data_files
        self.train_val_test_split = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.chunk_length = chunk_length
        self.shuffle_train = shuffle_train

        self.batch_size = batch_size

        self.trainset: Dataset | None = None
        self.validset: Dataset | None = None
        self.validset_full: Dataset | None = None
        self.testset: Dataset | None = None

    def prepare_data(self) -> None:
        # Nothing to generate/cached anymore; just validate files exist
        _ = _expand_files(self.data_files)

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.trainset is None:
            full_database = DedalusDynamicsDatabase(self.data_files)

            train_db, valid_db, test_db = random_split(full_database, self.train_val_test_split)

            self.trainset = DedalusDynamicsDataset(train_db, chunk_length=self.chunk_length)
            self.validset = DedalusDynamicsDataset(valid_db, chunk_length=self.chunk_length)
            self.validset_full = DedalusDynamicsDataset(valid_db, chunk_length=-1)
            self.testset = DedalusDynamicsDataset(test_db, chunk_length=self.chunk_length)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=torch.stack,
            shuffle=self.shuffle_train,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        valid_dl = DataLoader(
            dataset=self.validset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=torch.stack,
            shuffle=False,
        )
        valid_dl_full = DataLoader(
            dataset=self.validset_full,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=torch.stack,
            shuffle=False,
        )
        return CombinedLoader({"valid": valid_dl, "valid_full": valid_dl_full}, mode="sequential")

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=torch.stack,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return self.test_dataloader()

    def teardown(self, stage: str | None = None) -> None:
        del self.trainset
        del self.validset
        del self.validset_full
        del self.testset
