#!/usr/bin/env bash
# Reset (clear outputs) and execute every Jupyter notebook in the listed subfolders.
# Run from the repo root or from notebooks/ — the script resolves its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUBFOLDERS=(
    # prediction
    potential_evolution
    potential_scatter
    component_scaling
    # potential_variation_dt_scaling
    # prediction_table
)

# Use the pixi-kernel Python kernel: it auto-discovers pixi.toml/pyproject.toml
# from the notebook directory and launches in the pixi default environment.
# See https://pixi.prefix.dev/latest/integration/editor/jupyterlab/
KERNEL_NAME="pixi-kernel-python3"

if ! pixi run jupyter kernelspec list 2>/dev/null | awk 'NR>1 {print $1}' \
        | grep -qx "${KERNEL_NAME}"; then
    echo "error: kernel '${KERNEL_NAME}' not found — install pixi-kernel in the pixi env" >&2
    exit 1
fi
echo "==> using kernel: ${KERNEL_NAME}"

run_notebook() {
    local nb="$1"
    echo "==> $nb"
    pixi run jupyter nbconvert --clear-output --inplace "$nb"
    pixi run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" "$nb"
}

for sub in "${SUBFOLDERS[@]}"; do
    if [[ ! -d "$sub" ]]; then
        echo "[skip] $sub (not a directory)"
        continue
    fi
    shopt -s nullglob
    nbs=("$sub"/*.ipynb)
    shopt -u nullglob
    if (( ${#nbs[@]} == 0 )); then
        echo "[skip] $sub (no .ipynb)"
        continue
    fi
    for nb in "${nbs[@]}"; do
        run_notebook "$nb"
    done
done

echo "All notebooks reset and executed."
