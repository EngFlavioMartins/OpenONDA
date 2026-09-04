#!/usr/bin/env bash
# Install ParaView into the canonical OpenONDA environment so its Python ABI
# is resolved together with the rest of the runtime.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/install/install_paraview.sh

Install the optional ParaView application into the OpenONDA Conda environment.
Set OPENONDA_CONDA_ENV to select a non-default environment name.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

CONDA_ENV="${OPENONDA_CONDA_ENV:-OpenONDA}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found. Run scripts/install/install_conda.sh first." >&2
    exit 1
fi

conda install --name "$CONDA_ENV" --channel conda-forge --override-channels --yes paraview
conda run -n "$CONDA_ENV" paraview --version

echo "ParaView is installed in conda env '$CONDA_ENV'."
