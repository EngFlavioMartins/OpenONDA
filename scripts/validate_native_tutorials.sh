#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openonda-native-tutorials.XXXXXX")"

cleanup() {
    if [[ "${OPENONDA_KEEP_SMOKE:-0}" == "1" ]]; then
        echo "Keeping smoke outputs at $WORK_DIR"
    else
        rm -rf -- "$WORK_DIR"
    fi
}
trap cleanup EXIT

(
    cd "$WORK_DIR"
    env PYTHONNOUSERSITE=1 python - <<'PY'
import pathlib
import openonda
import openonda.coupler
import openonda.fvm
import openonda.vpm

package = pathlib.Path(openonda.__file__).resolve()
print(f"Validating OpenONDA {openonda.__version__} from {package}")
PY
)

run_case() {
    local name="$1"
    local end_time="${2:-}"
    local source="$REPO_ROOT/tutorials/coupled_FVM_VPM/$name"
    local target="$WORK_DIR/$name"
    mkdir -p "$target"
    rsync -a \
        --exclude='__pycache__/' \
        --exclude='.pytest_cache/' \
        --exclude='*.log' \
        --exclude='*.pyc' \
        --exclude='/figures/' \
        --exclude='/referenceFlow/' \
        --exclude='/samples/' \
        --exclude='/solution/' \
        "$source/" "$target/"
    echo "Running installed-package smoke: $name"
    if [[ -n "$end_time" ]]; then
        (
            cd "$target"
            env PYTHONNOUSERSITE=1 OPENONDA_SMOKE=1 OPENONDA_T_END="$end_time" \
                OPENONDA_PROCESSING_UNIT=CPU ./allrun.sh
        )
    else
        (
            cd "$target"
            env PYTHONNOUSERSITE=1 OPENONDA_SMOKE=1 OPENONDA_PROCESSING_UNIT=CPU ./allrun.sh
        )
    fi
}

run_fvm_smoke() {
    local source="$REPO_ROOT/tutorials/FVM/taylorGreen"
    local target="$WORK_DIR/taylorGreen"
    rsync -a \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='/figures/' \
        --exclude='/solution/' \
        "$source/" "$target/"
    echo "Running standalone FVM smoke: Taylor-Green vortex"
    (
        cd "$target"
        env PYTHONNOUSERSITE=1 python taylorGreen_setup.py \
            --n 8 --nu 0.1 --dt 0.005 --end-time 0.01
    )
    env PYTHONNOUSERSITE=1 python - "$target/solution/history.csv" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 3:
    raise SystemExit(f"Expected three Taylor-Green history rows, found {len(rows)}")
final = {key: float(value) for key, value in rows[-1].items() if key != "step"}
if not all(math.isfinite(value) for value in final.values()):
    raise SystemExit("Taylor-Green history contains non-finite values")
if final["continuity_max"] > 1.0e-10 or final["velocity_l2_error"] > 1.0e-2:
    raise SystemExit(f"Taylor-Green acceptance limits failed: {final}")
print(
    "PASS: standalone FVM Taylor-Green run completed with "
    f"velocity L2 error={final['velocity_l2_error']:.3e} and "
    f"continuity={final['continuity_max']:.3e}."
)
PY
}

run_vpm_smoke() {
    local source="$REPO_ROOT/tutorials/VPM/lambOseenVortex"
    local target="$WORK_DIR/lambOseenVortex"
    rsync -a \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='/figures/' \
        --exclude='/solution/' \
        "$source/" "$target/"
    echo "Running standalone VPM smoke: Lamb-Oseen vortex"
    (
        cd "$target"
        env PYTHONNOUSERSITE=1 python vortex_setup.py \
            --gamma1 1.0 --gamma2 0.0 --schemes cs --num-steps 2 \
            --length 2 --spacing-factor 0.3 --processing-unit CPU \
            --backup-frequency 1 --solution-dir "$target/solution"
    )
    env PYTHONNOUSERSITE=1 python - "$target/solution" <<'PY'
import pathlib
import sys

import h5py
import numpy as np

snapshots = sorted(pathlib.Path(sys.argv[1]).glob("vpm_vortex_cs_*.h5"))
if len(snapshots) != 2:
    raise SystemExit(f"Expected two VPM snapshots, found {len(snapshots)}")
with h5py.File(snapshots[-1], "r") as handle:
    invalid = []

    def check(name, item):
        if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
            if not np.all(np.isfinite(item[...])):
                invalid.append(name)

    handle.visititems(check)
if invalid:
    raise SystemExit(f"VPM snapshot contains non-finite datasets: {invalid}")
print("PASS: standalone VPM Lamb-Oseen run advanced two steps with finite snapshots.")
PY
}

run_fvm_smoke
run_vpm_smoke

if [[ "${1:-}" == "--extended" ]]; then
    run_case cubeFlow 0.25
    run_case cylinderSheddingFlow 0.50
    run_case naca4412Flow 0.40
elif [[ $# -eq 0 ]]; then
    run_case cubeFlow
    run_case cylinderSheddingFlow
    run_case naca4412Flow
else
    echo "Usage: scripts/validate_native_tutorials.sh [--extended]" >&2
    exit 2
fi

echo "Standalone FVM, standalone VPM, and native FVM-VPM tutorial smokes passed."
