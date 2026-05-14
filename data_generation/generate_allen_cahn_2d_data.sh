#!/bin/bash

# 2D Allen-Cahn dataset generation
# Equation: u_t = Δu + (1/ε²)(u - u³), periodic on [0, 2π]^2.
# Default settings give ~41 frames per trajectory at 128x128 resolution.

cmd_args=(
    python data_generation/allen_cahn_2d/generate.py
    -n 1000           # number of trajectories
    -r 0              # random seed
    --eps 0.1         # phase-field thickness
    --Nx 128          # grid points in x
    --Ny 128          # grid points in y
    --T 0.1           # total simulation time
    --dt 4e-5         # time step (SBDF2; nonlinear-explicit limit ~ε²)
    --save_every 25   # 0.1 / 4e-5 / 25 ≈ 100 -> 101 frames
)

pixi run "${cmd_args[@]}"
