#!/bin/bash

# Paper-facing FENE launcher.
# FENE dataset generation (kdv.ipynb accurate_test parameters)
# eps ≈ 0.03 (Lx=2pi, N_chain=209), c=1, T=1.0
# Run time: ~minutes for small n, ~hours for n=10000

cmd_args=(
    python data_generation/fene/generate.py
    -n 1000           # number of samples
    -r 0              # random seed
    --N_chain 209     # chain sites (eps = 2pi/209 ≈ 0.03)
    --T 0.5           # total slow time
    --dt_kdv 5e-4     # slow-time step
    --save_every 10    # save every 10 slow steps -> 51 frames
    --chain_substeps 1  # auto-increased for CFL stability
)

pixi run "${cmd_args[@]}"
