#!/bin/bash

# Paper-facing FPUT launcher.
# Define arguments in an array
cmd_args=(
    python data_generation/fput/generate.py
    -n 1000          # number of samples
    -r 0             # random seed
    --N_kdv 256      # KdV output grid points
    --N_chain 128    # chain sites; eps = Lx/N_chain ≈ 0.049
    --chain_substeps 50   # substeps per slow-time step (increase if unstable)
    --T 1.0          # total slow time
    --save_every 10  # save every 10 slow-time steps → 101 frames
)

# Expand the array into the command
pixi run "${cmd_args[@]}"
