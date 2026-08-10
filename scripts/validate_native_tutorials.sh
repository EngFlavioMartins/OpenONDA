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

echo "All native FVM-VPM tutorial smoke cases passed."
