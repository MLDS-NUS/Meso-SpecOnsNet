# Universal Dynamics Data Framework

This directory contains a universal framework for managing trajectory data across different dynamical systems (SIR, VPFP1D, etc.).

## Architecture Overview

The framework provides:

1. **Universal base classes** for configuration, dataset, and datamodule
2. **Shared utilities** for data I/O and config management
3. **Dynamics-specific implementations** that inherit from the universal base

### Directory Structure

```txt
src/data/
├── base_config.py              # DynamicsConfig base class
├── config_utils.py             # Shared config loading utilities
├── data_utils.py               # Universal HDF5 save/load functions
├── universal_datamodule.py     # DynamicsDatabase, DynamicsDataset, DynamicsDataModule
├── universal_cache_manager.py  # Universal cache management CLI tool
│
├── sir_config.py               # SIR-specific config (inherits from DynamicsConfig)
├── sir_datamodule.py           # SIR-specific datamodule (legacy, still works)
├── generate_sir_data.py        # SIR data generation utilities
├── cache_manager.py            # SIR-specific cache manager (legacy)
│
├── pde/
│   ├── vpfp1d/
│   │   ├── config.py           # VPFP1D-specific config
│   │   ├── simulator.py        # VPFP1D simulator
│   │   ├── data_utils.py       # VPFP1D data generation
│   │   └── __init__.py
│   └── pde_simulator_base.py
│
└── particle/
    └── ...
```

## Adding a New Dynamics Type

To add support for a new dynamics type (e.g., "my_dynamics"):

### 1. Create a Config Class

Create `src/data/[category]/my_dynamics/config.py`:

```python
from dataclasses import dataclass
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.base_config import DynamicsConfig
from src.data.config_utils import LazyConfigDict, get_config, list_available_configs


@dataclass
class MyDynamicsConfig(DynamicsConfig):
    # Your dynamics-specific parameters
    param1: float = 1.0
    param2: int = 100

    # Common parameters
    n_trajectories: int = 1000
    batch_size: int = 32
    cache_dir: str = "data/trajectories"
    cache_name: str | None = None
    force_regenerate: bool = False

    @property
    def generation_params(self) -> dict:
        """Return parameters that affect data generation."""
        return {
            "param1": self.param1,
            "param2": self.param2,
            "n_trajectories": self.n_trajectories,
            # ... other params
        }

    @property
    def dynamics_type(self) -> str:
        return "my_dynamics"


# Convenience functions
def get_my_dynamics_config(name: str) -> MyDynamicsConfig:
    return get_config(MyDynamicsConfig, name, "my_dynamics")


def list_my_dynamics_configs() -> list[str]:
    return list_available_configs("my_dynamics")


CONFIGS = LazyConfigDict(MyDynamicsConfig, "my_dynamics")
```

### 2. Create Data Utilities

Create `src/data/[category]/my_dynamics/data_utils.py`:

```python
import torch
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.data_utils import load_trajectories_hdf5, save_trajectories_hdf5
from src.data.[category].my_dynamics.config import MyDynamicsConfig
from src.data.[category].my_dynamics.simulator import MyDynamicsSimulator


def generate_my_dynamics_trajectories(config: MyDynamicsConfig) -> dict[str, torch.Tensor]:
    """Generate trajectory data for my dynamics."""
    simulator = MyDynamicsSimulator(config)

    # Your generation logic here
    trajectories = simulator.generate(...)

    return {"full": trajectories}


def save_my_dynamics_data(data_dict: dict[str, torch.Tensor], save_path: str) -> None:
    """Save data using universal utilities."""
    extra_metadata = {
        # Add any dynamics-specific metadata
    }
    save_trajectories_hdf5(data_dict, save_path, dynamics_type="my_dynamics", **extra_metadata)


def load_my_dynamics_data(data_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load data using universal utilities."""
    return load_trajectories_hdf5(data_path)
```

### 3. Use the Universal DataModule

```python
from src.data.universal_datamodule import DynamicsDataModule
from src.data.[category].my_dynamics.config import MyDynamicsConfig
from src.data.[category].my_dynamics.data_utils import (
    generate_my_dynamics_trajectories,
    load_my_dynamics_data,
    save_my_dynamics_data
)

# Create config
config = MyDynamicsConfig(n_trajectories=1000, param1=2.0)

# Create datamodule
datamodule = DynamicsDataModule(
    config=config,
    generator_fn=generate_my_dynamics_trajectories,
    load_fn=load_my_dynamics_data,
    save_fn=save_my_dynamics_data,
    split_ratio=(0.8, 0.1, 0.1),
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    chunk_length=10,
    data_mode="full",  # or whatever key you use in data_dict
)

# Use with PyTorch Lightning
datamodule.prepare_data()
datamodule.setup()
train_loader = datamodule.train_dataloader()
```

### 4. Create Config YAML Files (Optional)

Create `configs/data/my_dynamics/default.yaml`:

```yaml
# My Dynamics Configuration
param1: 1.0
param2: 100

n_trajectories: 1000
batch_size: 32

cache_dir: "data/trajectories"
cache_name: "my_dynamics_default"
force_regenerate: false
```

## Example Usage

### Using SIR (Legacy and New Way)

#### Legacy Way (still works)

```python
from src.data import SIRDataModule, SIRTrajConfig

config = SIRTrajConfig(grid_size=128, n_trajectories=100)
datamodule = SIRDataModule(config, split_ratio=(0.8, 0.1, 0.1), ...)
```

#### New Universal Way

```python
from src.data import DynamicsDataModule
from src.data.sir_config import SIRTrajConfig
from src.data.generate_sir_data import generate_sir_trajectories, load_sir_data, save_sir_data

config = SIRTrajConfig(grid_size=128, n_trajectories=100)
datamodule = DynamicsDataModule(
    config=config,
    generator_fn=generate_sir_trajectories,
    load_fn=load_sir_data,
    save_fn=save_sir_data,
    split_ratio=(0.8, 0.1, 0.1),
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    chunk_length=10,
    data_mode="meso",
)
```

### Using VPFP1D

```python
from src.data import DynamicsDataModule
from src.data.pde.vpfp1d import (
    VPFP1DConfig,
    generate_vpfp1d_trajectories,
    load_vpfp1d_data,
    save_vpfp1d_data
)

config = VPFP1DConfig(Nx=256, Nv=256, n_trajectories=100)
datamodule = DynamicsDataModule(
    config=config,
    generator_fn=generate_vpfp1d_trajectories,
    load_fn=load_vpfp1d_data,
    save_fn=save_vpfp1d_data,
    split_ratio=(0.8, 0.1, 0.1),
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    chunk_length=20,
    data_mode="full",
)
```

## Cache Management

### Universal Cache Manager

The universal cache manager is the recommended tool for managing datasets across all dynamics types (SIR, VPFP1D, etc.).

#### Overview

The cache manager provides four main commands:

- **list**: View all cached datasets with their validation status
- **configs**: List available configurations for each dynamics type
- **generate**: Generate datasets from YAML configurations
- **clean**: Remove invalid or corrupted cached datasets

#### Basic Commands

```bash
# List all cached datasets (all dynamics types)
python src/data/universal_cache_manager.py list --cache-dir data/trajectories

# List available configurations for all dynamics types
python src/data/universal_cache_manager.py configs

# List available configurations for a specific dynamics type
python src/data/universal_cache_manager.py configs --type vpfp1d
python src/data/universal_cache_manager.py configs --type sir

# Clean invalid datasets (dry run - shows what would be deleted)
python src/data/universal_cache_manager.py clean --cache-dir data/trajectories

# Clean invalid datasets (actually delete files)
python src/data/universal_cache_manager.py clean --cache-dir data/trajectories --no-dry-run
```

#### Generating VPFP1D Data

**Step 1: Check available VPFP1D configurations**

```bash
python src/data/universal_cache_manager.py configs --type vpfp1d
```

This will show all available VPFP1D configurations defined in `configs/data/vpfp1d/*.yaml`, including:

- Grid dimensions (Nx, Nv)
- Time parameters (maxT, dt)
- Number of trajectories
- Initial field type
- Cache name

**Step 2: Generate dataset from a specific configuration**

```bash
# Generate data using the 'demo' config
python src/data/universal_cache_manager.py generate vpfp1d demo

# Generate multiple configs at once
python src/data/universal_cache_manager.py generate vpfp1d demo small large

# Force regeneration even if valid cache exists
python src/data/universal_cache_manager.py generate vpfp1d demo --force
```

**Step 3: Verify the generated dataset**

```bash
# List all cached datasets to verify generation
python src/data/universal_cache_manager.py list
```

The output will show:

- File name (e.g., `vpfp1d/demo`)
- Dynamics type (`vpfp1d`)
- Size in MB
- Validation status (`VALID` or `INVALID`)
- Configuration hash
- Any error messages if validation failed

#### Example: Complete VPFP1D Workflow

```bash
# 1. See what VPFP1D configs are available
python src/data/universal_cache_manager.py configs --type vpfp1d

# Example output:
# Available VPFP1D configurations:
# ========================================
#
# 🔧 demo:
#    Grid: 256x256
#    Max time: 10.0
#    Time step: 0.1
#    Trajectories: 100
#    Init field: double_gaussian
#    Cache name: vpfp1d_demo
#
# 🔧 small:
#    Grid: 128x128
#    Max time: 5.0
#    ...

# 2. Generate the demo dataset
python src/data/universal_cache_manager.py generate vpfp1d demo

# This will:
# - Load the config from configs/data/vpfp1d/demo.yaml
# - Check if valid cached data exists
# - Generate data if needed (may take several minutes)
# - Save to data/trajectories/vpfp1d/demo/data.h5
# - Save config metadata to data/trajectories/vpfp1d/demo/config.json

# 3. Verify the dataset was created successfully
python src/data/universal_cache_manager.py list

# 4. (Optional) If you modify the config and want to regenerate
python src/data/universal_cache_manager.py generate vpfp1d demo --force
```

#### Creating Custom VPFP1D Configurations

To create a new VPFP1D configuration:

1. Create a new YAML file in `configs/data/vpfp1d/my_config.yaml`:

```yaml
# VPFP1D Configuration
Nx: 256                      # Spatial grid points
Nv: 256                      # Velocity grid points
xL: 10.0                     # Spatial domain length
vL: 10.0                     # Velocity domain length
maxT: 10.0                   # Maximum simulation time
dt: 0.1                      # Time step
nu: 0.1                      # Collision frequency
init_field: double_gaussian  # Initial condition type

n_trajectories: 100          # Number of trajectories to generate
batch_size: 16               # Batch size for training

cache_dir: "data/trajectories"
cache_name: "vpfp1d_my_config"
force_regenerate: false
```

2. Generate the dataset:

```bash
python src/data/universal_cache_manager.py generate vpfp1d my_config
```

#### Understanding Cache Validation

The cache manager automatically validates cached datasets by:

1. Checking if `data.h5` file exists
2. Checking if `config.json` metadata exists
3. Computing a hash of generation parameters
4. Comparing cached config hash with current config hash
5. Marking dataset as VALID only if hashes match

If you modify a config file (e.g., change `Nx` or `n_trajectories`), the hash will change and the cache will be marked INVALID. Use the `--force` flag or clean invalid datasets and regenerate.

#### Cache Directory Structure

```txt
data/trajectories/
├── sir/
│   ├── default/
│   │   ├── data.h5
│   │   └── config.json
│   └── small-test/
│       ├── data.h5
│       └── config.json
└── vpfp1d/
    ├── demo/
    │   ├── data.h5
    │   └── config.json
    └── large/
        ├── data.h5
        └── config.json
```

### Legacy SIR Cache Manager (still works)

```bash
# List SIR datasets
python src/data/cache_manager.py list

# Generate specific SIR configs
python src/data/cache_manager.py generate small-test medium-train

# Clean invalid SIR datasets
python src/data/cache_manager.py clean --no-dry-run
```

## Key Features

1. **Automatic Caching**: Data is automatically cached based on config hash
2. **Cache Validation**: Cached data is validated against config parameters
3. **Flexible Data Modes**: Support for multiple data representations (e.g., micro/meso for SIR)
4. **Universal I/O**: Shared HDF5 save/load functions
5. **Config Management**: YAML-based config loading with lazy loading support
6. **Backward Compatibility**: Legacy SIR code still works

## Benefits

- **Consistency**: All dynamics follow the same pattern
- **Reusability**: Shared utilities reduce code duplication
- **Maintainability**: Changes to universal components benefit all dynamics
- **Extensibility**: Easy to add new dynamics types
- **Type Safety**: Strong typing with dataclasses and type hints
