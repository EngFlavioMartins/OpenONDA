#!/bin/sh
# Generate figures from solution/ (forces, VPM diagnostics, hybrid wake).
# Reads only what cube_setup.py wrote — safe to re-run any time after a run.
cd "$(dirname "$0")" || exit 1

PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null \
        || conda run -n OpenONDA which python 2>/dev/null \
        || command -v python3 || command -v python)"

# Matplotlib needs a writable config dir in headless/CI environments.
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

FORMAT="png"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --format) FORMAT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH}"

for script in plot_forces plot_diagnostics plot_wake; do
    echo "=== ${script} ==="
    "$PYTHON" "assets/${script}.py" --format "$FORMAT" || echo "  (${script} failed; skipping)"
done

echo "Figures written to figures/"
