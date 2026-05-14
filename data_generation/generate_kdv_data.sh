#!/bin/bash

# Define arguments in an array
cmd_args=(
    python data_generation/kdv.py
    --num_pde=1000   # number of trajectories
    -r 0             # random seed
)

# Expand the array into the command
pixi run "${cmd_args[@]}"
