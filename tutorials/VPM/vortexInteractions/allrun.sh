#!/usr/bin/env bash
# Vortex-ring interactions — strength-relaxation comparison.
#
# This script reruns the intended four-case matrix at the same initial particle
# concentration. Every run is LES with transposed vortex stretching; the only
# stabilization difference is strength relaxation on/off:
#
#   LEAPFROG, Γ1 = Γ2 = +π:
#     1. leapfrog_les      2. leapfrog_les_isr
#
#   HEAD-ON COLLISION, Γ1 = +π, Γ2 = -π:
#     3. collide_les       4. collide_les_isr
#
# Every case writes:
#   - stability_metrics.csv at every step
#   - flow_integrals.csv at LOGGING_FREQUENCY
#   - energy_budget.csv at ENERGY_AUDIT_FREQUENCY
#
# Existing result directories for these four case names are deleted before their
# rerun so each directory contains one clean, comparable realization.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$(conda run -n OpenONDA which python 2>/dev/null \
          || command -v python3 \
          || command -v python)"

GAMMA_PI="3.14159265358979"

RUN_ROOT="${RUN_ROOT:-solution}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"
PARTICLE_SPACING="${PARTICLE_SPACING:-0.030}"

# Leapfrog uses CS with a smaller step than the DVH-pinned collision cases.
LF_DT="${LF_DT:-0.02}"
LF_STEPS="${LF_STEPS:-600}"

# Collision uses DVH; rings_setup.py pins the actual step to the DVH diffusion
# time when DVH is selected. 120 steps covers t≈12.4 s at h=0.025.
COLLIDE_DT="${COLLIDE_DT:-0.103}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-120}}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
ENERGY_AUDIT_FREQUENCY="${ENERGY_AUDIT_FREQUENCY:-1}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"
echo "Results root: $RUN_ROOT"
echo "Figures root: $FIGURES_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Energy-audit frequency: every $ENERGY_AUDIT_FREQUENCY step(s)"

RESULTS=()

run_case() {
    local label="$1"; shift
    echo ""
    echo "========================================================================"
    echo "${label}"
    echo "========================================================================"

    local case_name=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--name" && $((i+1)) -lt ${#args[@]} ]]; then
            case_name="${args[i+1]}"
            break
        fi
    done

    if [[ -n "$case_name" ]]; then
        rm -rf "$RUN_ROOT/$case_name"
    fi

    local rc=0
    "$PYTHON" rings_setup.py --output-root "$RUN_ROOT" "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "-> ${label} complete."
        RESULTS+=("PASS — ${label}")
    else
        echo "*** ${label} exited with code ${rc}; continuing ***" >&2
        RESULTS+=("FAIL (exit ${rc}) — ${label}")
    fi
}

# ---------------------------------------------------------------------------
# LEAPFROGGING: Γ1 = Γ2 = +π
# ---------------------------------------------------------------------------

run_case "1/4 leapfrog_les — LES transposed" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name leapfrog_les

run_case "2/4 leapfrog_les_isr — LES transposed + strength relaxation" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs \
    --relaxation blend --relaxation-rate 1.5 --deconv 1 \
    --device vulkan \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name leapfrog_les_isr

# ---------------------------------------------------------------------------
# HEAD-ON COLLISION: Γ1 = +π, Γ2 = -π
# ---------------------------------------------------------------------------

run_case "3/4 collide_les — LES transposed" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$COLLIDE_DT" --num-steps "$COLLIDE_STEPS" \
    --viscous dvh \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name collide_les

run_case "4/4 collide_les_isr — LES transposed + strength relaxation" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$COLLIDE_DT" --num-steps "$COLLIDE_STEPS" \
    --viscous dvh \
    --relaxation blend --relaxation-rate 1.5 --deconv 1 \
    --device vulkan \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name collide_les_isr

echo ""
echo "========================================================================"
echo "Run summary"
echo "========================================================================"
n_pass=0
n_fail=0
for entry in "${RESULTS[@]}"; do
    echo "  ${entry}"
    if [[ "${entry}" == PASS* ]]; then
        (( n_pass++ ))
    else
        (( n_fail++ ))
    fi
done
echo ""
echo "  ${n_pass} passed, ${n_fail} failed"
echo ""

if [[ -x ./allplot.sh ]]; then
    echo "Generating comparison figures..."
    ./allplot.sh --solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT" || true
fi

[[ $n_fail -eq 0 ]]
