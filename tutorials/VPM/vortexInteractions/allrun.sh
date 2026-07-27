#!/usr/bin/env bash
# Six vortex-ring interactions: 3 methods x 2 interaction families.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
    || command -v python3 \
    || command -v python)}"

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
DEFAULT_METHODS="baseline les les_stabilized"
DEFAULT_FAMILIES="leapfrog collide"
METHODS="${METHODS:-$DEFAULT_METHODS}"
RUN_FAMILIES="${RUN_FAMILIES:-$DEFAULT_FAMILIES}"
CLEAN_ALL="${CLEAN_ALL:-0}"
ALLOW_UNDERRESOLVED="${ALLOW_UNDERRESOLVED:-0}"
RUN_PLOTS="${RUN_PLOTS:-1}"

if [[ "$CLEAN_ALL" == "1" ]]; then
    ./allclean.sh --all
elif [[ "$CLEAN_ALL" != "0" ]]; then
    echo "CLEAN_ALL must be 0 or 1, got: $CLEAN_ALL" >&2
    exit 2
fi

if [[ "$ALLOW_UNDERRESOLVED" != "0" && "$ALLOW_UNDERRESOLVED" != "1" ]]; then
    echo "ALLOW_UNDERRESOLVED must be 0 or 1, got: $ALLOW_UNDERRESOLVED" >&2
    exit 2
fi
if [[ "$RUN_PLOTS" != "0" && "$RUN_PLOTS" != "1" ]]; then
    echo "RUN_PLOTS must be 0 or 1, got: $RUN_PLOTS" >&2
    exit 2
fi

STAGING_ROOT="$RUN_ROOT/.running"
mkdir -p "$RUN_ROOT" "$FIGURES_ROOT" "$STAGING_ROOT"

echo "Results root: $RUN_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Processing unit: $PROCESSING_UNIT"
echo "Device memory fraction: $DEVICE_MEMORY_FRACTION"
echo "Baseline: DNS, Gaussian, treecode, fractional RK3, transposed stretching, CS"
echo "LES: Smagorinsky LES with the same baseline numerical core"
echo "LES + stabilized: LES, Winckelmans, direct, coupled RK2, conservative stretching, CS"
echo "Methods: $METHODS"
echo "Families: $RUN_FAMILIES"
echo "Transactional output: an existing case is replaced only after its rerun reaches a terminal status"

run_case() {
    local label="$1"
    local case_name="$2"
    shift 2

    echo ""
    echo "========================================================================"
    echo "$label"
    echo "========================================================================"

    local staged_case="$STAGING_ROOT/$case_name"
    local final_case="$RUN_ROOT/$case_name"
    local command=(
        "$PYTHON" rings_setup.py
        --output-root "$STAGING_ROOT"
        --name "$case_name"
    )
    if [[ "$ALLOW_UNDERRESOLVED" == "1" ]]; then
        command+=(--allow-underresolved)
    fi
    command+=("$@")
    rm -rf -- "$staged_case"

    if ! "${command[@]}"; then
        echo "ERROR: $case_name crashed. Existing results were preserved." >&2
        if [[ -f "$staged_case/$case_name.log" ]]; then
            tail -80 "$staged_case/$case_name.log" >&2
        fi
        return 1
    fi

    local manifest="$staged_case/run_manifest.json"
    if [[ ! -s "$manifest" ]] || ! grep -Eq \
        '"status": "(completed|terminated_nonphysical|rejected_physical_contract)"' "$manifest"; then
        echo "ERROR: $case_name did not record a valid terminal status." >&2
        echo "Existing results were preserved; staged output remains at $staged_case" >&2
        return 1
    fi

    local case_status
    case_status="$(sed -n 's/.*"status": "\\([^"]*\\)".*/\\1/p' "$manifest")"
    if [[ "$case_status" != "completed" && -d "$final_case" ]]; then
        local failed_root="$RUN_ROOT/.failed"
        local failed_case="$failed_root/${case_name}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$failed_root"
        mv -- "$staged_case" "$failed_case"
        echo "WARNING: rerun ended with status=$case_status." >&2
        echo "Preserved existing $final_case; rejected rerun is at $failed_case" >&2
        return 0
    fi

    rm -rf -- "$final_case"
    mv -- "$staged_case" "$final_case"
    grep -E '"status"|"completed_steps"|"requested_steps"' "$final_case/run_manifest.json"
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

rmdir "$STAGING_ROOT" 2>/dev/null || true

if [[ "$RUN_PLOTS" == "1" && -x ./allplot.sh ]]; then
    PLOT_ARGS=(--solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT")
    if [[ "$METHODS" != "$DEFAULT_METHODS" || "$RUN_FAMILIES" != "$DEFAULT_FAMILIES" ]]; then
        PLOT_ARGS+=(--allow-partial)
    fi
    ./allplot.sh "${PLOT_ARGS[@]}"
fi
