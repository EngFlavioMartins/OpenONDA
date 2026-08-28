#!/usr/bin/env bash
# Resource-safe end-to-end validation. Large FVM jobs are serial by design;
# independent static checks and documentation work can run concurrently.
set -euo pipefail

cd "$(dirname "$0")"
smoke="${OPENONDA_SMOKE:-0}"

if [ "$smoke" = "1" ]; then
    OPENONDA_GRID=smoke python assets/audit_mesh_geometry.py
    OPENONDA_SMOKE=1 OPENONDA_GRID=smoke ./reference_flow/allrun.sh
    OPENONDA_SMOKE=1 OPENONDA_GRID=smoke ./allrun.sh
else
    ./reference_flow/allrun.sh
    ./allrun.sh
fi

python assets/check_run.py

if [ "$smoke" = "1" ]; then
    echo "Smoke validation passed through physical force/probe sampling."
else
    python assets/analyse_coupled_benchmark.py
    python assets/analyse_von_karman.py
    ./allplot.sh png
    ./allplot.sh pdf
    echo "Reference independence and coupled validation completed."
fi
