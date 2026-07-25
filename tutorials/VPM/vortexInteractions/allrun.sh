#!/usr/bin/env bash
# Vortex-ring interactions — LES stabilizer benchmark.
# Runs leapfrogging and head-on collision cases with the same LES/transposed/RK3
# solver core and conservative RK-stage safeguard. Only the named stabilization
# method changes between cases.
set -euo pipefail

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GAMMA_PI="3.14159265358979"

RUN_ROOT="${RUN_ROOT:-solution}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"
PARTICLE_SPACING="${PARTICLE_SPACING:-0.045}"

DT="${DT:-0.010}"
LF_DT="${LF_DT:-$DT}"
LF_STEPS="${LF_STEPS:-720}"

COLLIDE_DT="${COLLIDE_DT:-$DT}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-600}}"

VISCOUS="${VISCOUS:-cs}"
PROCESSING_UNIT="${PROCESSING_UNIT:-GPU}"
REMESH_PROCESSING_UNIT="${REMESH_PROCESSING_UNIT:-GPU}"
DEVICE_MEMORY_FRACTION="${DEVICE_MEMORY_FRACTION:-0.5}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
BLOWUP_CHECK_FREQUENCY="${BLOWUP_CHECK_FREQUENCY:-10}"
STABILIZATIONS="${STABILIZATIONS:-control les rvpm relax remesh projection split energy adaptive}"
RUN_FAMILIES="${RUN_FAMILIES:-leapfrog collide}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"

echo "Results root: $RUN_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Viscous scheme: $VISCOUS"
echo "Processing unit: $PROCESSING_UNIT"
echo "Remesh/projection processing unit: $REMESH_PROCESSING_UNIT"
echo "Device memory fraction: $DEVICE_MEMORY_FRACTION"
echo "Solver core: LES, transposed stretching, RK3 advection/stretching"
echo "Stabilizations: $STABILIZATIONS"
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
    for stabilization in $STABILIZATIONS; do
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

    for stabilization in $STABILIZATIONS; do
        index=$((index + 1))
        case_name="${family}_${stabilization}"
        case_processing_unit="$PROCESSING_UNIT"
        case "$stabilization" in
            remesh|projection)
                case_processing_unit="$REMESH_PROCESSING_UNIT"
                ;;
        esac
        run_case "$index/$total $case_name" "$case_name" \
            --gamma1 "$gamma1" --gamma2 "$gamma2" \
            --particle-spacing "$PARTICLE_SPACING" \
            --dt "$dt" --num-steps "$steps" \
            --viscous "$VISCOUS" \
            --processing-unit "$case_processing_unit" \
            --device-memory-fraction "$DEVICE_MEMORY_FRACTION" \
            --stabilization "$stabilization" \
            --backup-frequency "$BACKUP_FREQUENCY" \
            --logging-frequency "$LOGGING_FREQUENCY" \
            --blowup-check-frequency "$BLOWUP_CHECK_FREQUENCY"
    done
done

if [[ -x ./allplot.sh ]]; then
    ./allplot.sh --solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT"
fi
