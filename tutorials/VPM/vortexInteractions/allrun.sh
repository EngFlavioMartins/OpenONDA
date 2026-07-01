#!/usr/bin/env bash
# Vortex-ring interactions — LES ring-interaction benchmark.
# Runs leapfrogging and head-on collision cases, then generates comparison figures.
set -euo pipefail

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GAMMA_PI="3.14159265358979"

RUN_ROOT="${RUN_ROOT:-solution}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"
PARTICLE_SPACING="${PARTICLE_SPACING:-0.030}"

LF_DT="${LF_DT:-0.020}"
LF_STEPS="${LF_STEPS:-450}"
LF_VISCOUS="${LF_VISCOUS:-cs}"

COLLIDE_DT="${COLLIDE_DT:-0.060}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-210}}"
COLLIDE_VISCOUS="${COLLIDE_VISCOUS:-gbd}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
BLOWUP_CHECK_FREQUENCY="${BLOWUP_CHECK_FREQUENCY:-10}"
STRETCHING="${STRETCHING:-transposed}"
CASE_SUFFIX="${CASE_SUFFIX:-$STRETCHING}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"

echo "Results root: $RUN_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Viscous (leapfrog / collide): $LF_VISCOUS / $COLLIDE_VISCOUS"
echo "Stretching: $STRETCHING"

run_case() {
    local label="$1"
    local case_name="$2"
    shift 2

    echo ""
    echo "========================================================================"
    echo "$label"
    echo "========================================================================"

    rm -rf "$RUN_ROOT/$case_name"

    "$PYTHON" rings_setup.py \
        --output-root "$RUN_ROOT" \
        --name "$case_name" \
        "$@"
}

run_case "1/2 leapfrog_${CASE_SUFFIX}" "leapfrog_${CASE_SUFFIX}" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous "$LF_VISCOUS" \
    --stretching "$STRETCHING" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --blowup-check-frequency "$BLOWUP_CHECK_FREQUENCY"

run_case "2/2 collide_${CASE_SUFFIX}" "collide_${CASE_SUFFIX}" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$COLLIDE_DT" --num-steps "$COLLIDE_STEPS" \
    --viscous "$COLLIDE_VISCOUS" \
    --stretching "$STRETCHING" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --blowup-check-frequency "$BLOWUP_CHECK_FREQUENCY"

if [[ -x ./allplot.sh ]]; then
    ./allplot.sh --solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT"
fi
