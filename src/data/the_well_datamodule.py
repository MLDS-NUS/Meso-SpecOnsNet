"""
Dataset + DataModule for "The Well" datasets.

Wraps the `the_well.data.WellDataset` to provide a similar interface as
DedalusDynamicsDataModule, supporting trajectory chunking and train/val/test splits.

Reference: https://polymathic-ai.org/the_well/
"""

from __future__ import annotations

from typing import Any

import einops
import tensordict as td
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from the_well.data import WellDataset
from torch.utils.data import DataLoader, Dataset


class TheWellDynamicsDatabase(Dataset):
    """
    Wraps WellDataset to provide full trajectories.

    Each item is a TensorDict with shape: (nt, n_vars, nx, ny)
    matching the format expected by DedalusDynamicsDataset.
    """

    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        well_split_name: str,
        dtype: torch.dtype = torch.float32,
        n_input_steps: int | None = None,
        field_names: list[str] | None = None,
    ):
        """
        Args:
            well_base_path: Base path to The Well datasets.
            well_dataset_name: Name of the dataset (e.g., "shear_flow").
            well_split_name: Split name ("train", "valid", or "test").
            dtype: Torch dtype for the output tensors.
            n_input_steps: Number of input steps to request. If None, attempts
                to get almost full trajectory (199 steps for 200 timestep data).
            field_names: List of field names to include. If None, includes all fields.
        """
        self.well_base_path = well_base_path
        self.well_dataset_name = well_dataset_name
        self.well_split_name = well_split_name
        self.dtype = dtype
        self.field_names = field_names

        # We want full trajectories, so request max input steps
        # The WellDataset requires n_steps_output >= 1, so we use n_input + 1 = total
        self._n_input_steps = n_input_steps if n_input_steps is not None else 199

        self._dataset = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=well_split_name,
            full_trajectory_mode=False,
            n_steps_input=self._n_input_steps,
            n_steps_output=1,
            min_dt_stride=1,
            max_dt_stride=1,
            return_grid=False,
            use_normalization=False,
        )

        # Store metadata
        self.metadata = self._dataset.metadata
        self.n_spatial_dims = self._dataset.n_spatial_dims
        self.size_tuple = self._dataset.size_tuple

        # Get field names from metadata
        all_field_names = []
        for order in sorted(self.metadata.field_names.keys()):
            all_field_names.extend(self.metadata.field_names[order])
        self._all_field_names = all_field_names

        # Determine which field indices to keep
        if field_names is not None:
            self._field_indices = [all_field_names.index(fn) for fn in field_names]
        else:
            self._field_indices = list(range(len(all_field_names)))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> td.TensorDict:
        sample = self._dataset[index]

        # Concatenate input and output fields along time dimension
        # input_fields: (n_input, H, W, C), output_fields: (1, H, W, C)
        input_fields = sample["input_fields"]
        output_fields = sample["output_fields"]
        full_traj = torch.cat([input_fields, output_fields], dim=0)  # (nt, H, W, C)

        # Select specific fields if requested
        if self._field_indices != list(range(full_traj.shape[-1])):
            full_traj = full_traj[..., self._field_indices]

        # Convert to desired dtype
        full_traj = full_traj.to(dtype=self.dtype)

        # Rearrange to match Dedalus format: (nt, n_vars, nx, ny)
        # From: (nt, H, W, C) -> (nt, C, H, W)
        full_traj = einops.rearrange(full_traj, "Nt Nx Ny Nvar -> Nt Nvar Nx Ny")

        data = td.TensorDict(
            {"seq": full_traj},
            batch_size=full_traj.shape[0],
        )

        return data


class TheWellDynamicsDataset(Dataset):
    """Yields chunks of consecutive timesteps from a trajectory."""

    def __init__(self, database: Dataset, chunk_length: int):
        super().__init__()
        self.database = database
        self.chunk_length = chunk_length

    def __len__(self) -> int:
        return len(self.database)

    def __getitem__(self, index: int) -> td.TensorDict:
        traj = self.database[index]  # TensorDict with batch_size = nt

        if self.chunk_length == -1:
            return traj

        if 0 < self.chunk_length <= traj.shape[0]:
            s = torch.randint(0, traj.shape[0] - self.chunk_length + 1, (1,)).item()
            return traj[s : s + self.chunk_length]

        raise ValueError(f"Invalid chunk_length {self.chunk_length} for trajectory length {traj.shape[0]}")


class TheWellDynamicsDataModule(LightningDataModule):
    """
    Lightning DataModule for The Well datasets.

    Provides train/val/test dataloaders with trajectory chunking.
    """

    def __init__(
        self,
        dyn_type: str,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int,
        chunk_length: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_train: bool = True,
        n_input_steps: int | None = None,
        field_names: list[str] | None = None,
    ) -> None:
        """
        Args:
            dyn_type: Identifier for the dynamics type (for logging/config).
            well_base_path: Base path to The Well datasets.
            well_dataset_name: Name of the dataset (e.g., "shear_flow").
            batch_size: Batch size for dataloaders.
            chunk_length: Length of trajectory chunks (-1 for full trajectory).
            num_workers: Number of dataloader workers.
            pin_memory: Whether to pin memory in dataloaders.
            shuffle_train: Whether to shuffle training data.
            n_input_steps: Number of input steps for trajectory loading.
            field_names: List of field names to include. If None, includes all fields.
        """
        super().__init__()

        self.dyn_type = dyn_type
        self.well_base_path = well_base_path
        self.well_dataset_name = well_dataset_name
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.n_input_steps = n_input_steps
        self.field_names = field_names

        self.batch_size_per_device = batch_size

        self.trainset: Dataset | None = None
        self.validset: Dataset | None = None
        self.validset_full: Dataset | None = None
        self.testset: Dataset | None = None

    def prepare_data(self) -> None:
        # The Well datasets should already be downloaded
        pass

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by number of devices ({self.trainer.world_size})"
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if self.trainset is None:
            # Create databases for each split
            train_db = TheWellDynamicsDatabase(
                well_base_path=self.well_base_path,
                well_dataset_name=self.well_dataset_name,
                well_split_name="train",
                n_input_steps=self.n_input_steps,
                field_names=self.field_names,
            )
            valid_db = TheWellDynamicsDatabase(
                well_base_path=self.well_base_path,
                well_dataset_name=self.well_dataset_name,
                well_split_name="valid",
                n_input_steps=self.n_input_steps,
                field_names=self.field_names,
            )
            test_db = TheWellDynamicsDatabase(
                well_base_path=self.well_base_path,
                well_dataset_name=self.well_dataset_name,
                well_split_name="test",
                n_input_steps=self.n_input_steps,
                field_names=self.field_names,
            )

            self.trainset = TheWellDynamicsDataset(train_db, chunk_length=self.chunk_length)
            self.validset = TheWellDynamicsDataset(valid_db, chunk_length=self.chunk_length)
            self.validset_full = TheWellDynamicsDataset(valid_db, chunk_length=-1)
            self.testset = TheWellDynamicsDataset(test_db, chunk_length=self.chunk_length)

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


# dm = TheWellDynamicsDataModule(
#     dyn_type="shear_flow",
#     well_base_path="~/nfs/datasets/the_well/datasets",
#     well_dataset_name="shear_flow",
#     batch_size=4,
#     chunk_length=50,
#     num_workers=4,
#     pin_memory=True,
# )
