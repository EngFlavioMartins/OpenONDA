#!/usr/bin/env bash
# Vortex-ring interactions — baseline versus stabilized conservative method.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"

./allclean.sh

GAMMA_PI="3.14159265358979"

RUN_ROOT="${RUN_ROOT:-solution}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"
PARTICLE_SPACING="${PARTICLE_SPACING:-0.020}"

DT="${DT:-0.00254647908947033}"
LF_DT="${LF_DT:-$DT}"
LF_STEPS="${LF_STEPS:-2800}"

COLLIDE_DT="${COLLIDE_DT:-$DT}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-2400}}"

PROCESSING_UNIT="${PROCESSING_UNIT:-AUTO}"
DEVICE_MEMORY_FRACTION="${DEVICE_MEMORY_FRACTION:-0.5}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-100}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-20}"
GUARD_FREQUENCY="${GUARD_FREQUENCY:-1}"
METHODS="${METHODS:-baseline stabilized}"
RUN_FAMILIES="${RUN_FAMILIES:-leapfrog collide}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"

echo "Results root: $RUN_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Processing unit: $PROCESSING_UNIT"
echo "Device memory fraction: $DEVICE_MEMORY_FRACTION"
echo "Stabilized core: DNS, Winckelmans, direct, coupled RK2, conservative stretching, CS"
echo "Methods: $METHODS"
echo "Families: $RUN_FAMILIES"

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

total=0
for family in $RUN_FAMILIES; do
    for method in $METHODS; do
        total=$((total + 1))
    done
done

index=0
for family in $RUN_FAMILIES; do
    case "$family" in
        leapfrog)
            gamma1="$GAMMA_PI"
            gamma2="$GAMMA_PI"
            dt="$LF_DT"
            steps="$LF_STEPS"
            ;;
        collide)
            gamma1="$GAMMA_PI"
            gamma2="-$GAMMA_PI"
            dt="$COLLIDE_DT"
            steps="$COLLIDE_STEPS"
            ;;
        *)
            echo "Unknown family in RUN_FAMILIES: $family" >&2
            exit 1
            ;;
    esac

    for method in $METHODS; do
        index=$((index + 1))
        case_name="${family}_${method}"
        run_case "$index/$total $case_name" "$case_name" \
            --gamma1 "$gamma1" --gamma2 "$gamma2" \
            --particle-spacing "$PARTICLE_SPACING" \
            --dt "$dt" --num-steps "$steps" \
            --processing-unit "$PROCESSING_UNIT" \
            --device-memory-fraction "$DEVICE_MEMORY_FRACTION" \
            --method "$method" \
            --backup-frequency "$BACKUP_FREQUENCY" \
            --logging-frequency "$LOGGING_FREQUENCY" \
            --guard-frequency "$GUARD_FREQUENCY"
    done
done

if [[ -x ./allplot.sh ]]; then
    ./allplot.sh --solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT"
fi
