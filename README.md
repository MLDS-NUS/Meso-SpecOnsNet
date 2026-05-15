<div align="center">

# Hypothesis-driven construction of mesoscopic dynamics

Reproducibility repository for review purpose.

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.9+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![pixi](https://img.shields.io/badge/env-pixi-yellow)](https://pixi.sh/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg?labelColor=gray)](https://docs.astral.sh/ruff/)<br>
[![license](https://img.shields.io/badge/License-MIT-blue.svg?labelColor=gray)](LICENSE)

</div>

## Overview

This repository provides the code for data generation, network training, and visualization, reproducing the computational results for the
paper **Hypothesis-driven construction of mesoscopic dynamics**.

## Quick start

The installation requires a linux-64 platform with CUDA>=12.6

```bash
# clone this repository
# install pixi (https://pixi.prefix.dev/latest/)
# and then run:
pixi install
```

You may skip the data generation and model training steps by directly downloading the files from the [huggingface repository](https://huggingface.co/datasets/MLDS-NUS/Meso-SpecOnsNet). Put the downloaded files as they are under the repo's root.

## Data

We provide data generation scripts for all datasets used in the paper. Each script generates the corresponding dataset and saves it as HDF5 files under `data/`.

```bash
# DATASET_NAME = kdv, allen_cahn, fput, fene, allen_cahn_2d
bash data_generation/generate_{DATASET_NAME}_data.sh
```

## Training

Each experiment is launched through Hydra with keys of the form
`experiment=<dataset>/<variant>`.

```bash
# DATASET_NAME = kdv, ac, fput, fene, ac_2d
# MODEL_NAME = 's_onsagernet' (SpecOnsNet, ours), 'fno', 'onsagernet' ('onsagernet' is not available for ac_2d)
# MODEL_NAME = 'ac', 'ac_2d', 'kdv' for classical solver
# MODEL_NAME = 'res_onsagernet' (Res-OnsagerNet; 'res_onsagernet_2d' for ac_2d), 'ac_realV'/'kdv_realV'/'ac_2d_realV' (SpecOnsNet-V) for ablation study in SM
pixi run python src/train.py experiment={DATASET_NAME}/{MODEL_NAME}
```

## Figures and notebooks

You may find the jupyter notebooks to generate all figures under under `notebooks/`, and the output figures are under `figs/`.

Run the following command to clear outputs and execute all notebooks in sequence:

```bash
$ ./notebooks/reset_and_run_all.sh
```

The notebook runner clears outputs and executes notebooks in the configured
Pixi Jupyter kernel. It assumes the required HDF5 datasets and model artifacts
are available at the paths expected by the notebooks.
