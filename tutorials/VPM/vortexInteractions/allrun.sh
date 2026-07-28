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

# Widnall-resolving spacing.  The particle radius is 2h, and the initial field
# is anti-diffused to a core sqrt(a0^2 - (2h)^2) so the *represented* core is
# a0 = 0.1.  That anti-diffused core must stay wider than the spacing or the
# initial Gaussian is aliased and the perturbation degenerates into numerical
# noise -- which is why the old h=0.045 study had to run axisymmetric.
#
#   h       sigma0   anti-diffused core   aliased   a0/h   pts per mode-24 wave
#   0.045   0.090    0.0436               YES       2.2    5.8
#   0.036   0.072    0.0694               no        2.8    7.2
#   0.030   0.060    0.0800               no        3.3    8.7     <- used here
#   0.020   0.040    0.0917               no        5.0    13.1    (143k particles)
#
# h = 0.030 is the smallest spacing that both clears aliasing with margin and
# keeps the O(N^2) direct/conservative stabilized method tractable at ~33k
# particles.  h = 0.020 would satisfy the paper's h/a0 <= 0.2 but needs 143k.
PARTICLE_SPACING="${PARTICLE_SPACING:-0.030}"
DT="${DT:-0.0057295779513082}"          # 20 h^2 / Gamma0
LF_DT="${LF_DT:-$DT}"
LF_STEPS="${LF_STEPS:-2100}"            # t_end = 12.0  (t* = 37.7)

COLLIDE_DT="${COLLIDE_DT:-$DT}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-1750}}"   # t_end = 10.0  (t* = 31.4)

PROCESSING_UNIT="${PROCESSING_UNIT:-AUTO}"
DEVICE_MEMORY_FRACTION="${DEVICE_MEMORY_FRACTION:-0.5}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-50}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-25}"
GUARD_FREQUENCY="${GUARD_FREQUENCY:-25}"
DEFAULT_METHODS="baseline les les_stabilized"
DEFAULT_FAMILIES="leapfrog collide"
METHODS="${METHODS:-$DEFAULT_METHODS}"
RUN_FAMILIES="${RUN_FAMILIES:-$DEFAULT_FAMILIES}"
CLEAN_ALL="${CLEAN_ALL:-0}"
# h/a0 = 0.30 exceeds the paper's 0.2; the binding constraint here is aliasing
# of the anti-diffused core, which this spacing clears.
ALLOW_UNDERRESOLVED="${ALLOW_UNDERRESOLVED:-1}"
EPSILON_W="${EPSILON_W:-0.025}"         # Widnall centreline perturbation
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
echo "Time step: $DT"
echo "Processing unit: $PROCESSING_UNIT"
echo "Device memory fraction: $DEVICE_MEMORY_FRACTION"
echo "Baseline: DNS, Gaussian, treecode, fractional RK3, transposed stretching, CS"
echo "LES: Smagorinsky LES with the same baseline numerical core"
echo "LES + stabilized: LES Cs=0.20, Gaussian, direct, coupled implicit midpoint, conservative stretching, CS"
echo "Methods: $METHODS"
echo "Families: $RUN_FAMILIES"
echo "Widnall perturbation amplitude: $EPSILON_W"
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
    case_status="$(awk -F'"' '/"status":/ { print $4; exit }' "$manifest")"
    # 'rejected_physical_contract' is a legitimate terminal outcome, not a
    # failure: the stabilized method is defined to run only as far as it stays
    # admissible, and that run IS the result. Only a genuine blow-up is
    # quarantined in favour of whatever was there before.
    if [[ "$case_status" == "terminated_nonphysical" && -d "$final_case" ]]; then
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
            --epsilon-w "$EPSILON_W" \
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
