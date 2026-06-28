#!/usr/bin/env bash
# Vortex-ring interactions — Winckelmans/Pedrizzetti strength-relaxation study.
#
# The relaxation realigns each particle's vector strength Γ with the local
# representable vorticity direction, preserving |Γ| exactly:
#     Γ ← |Γ| · normalize((1−r)·Γ̂ + r·ω̂_local).
# r = RELAX_FACTOR is the tuning knob.  This is the only stabilizer kept; the
# split/router, ISR limiter and solenoidal-projection experiments were removed.
#
# Four cases — bare LES vs relaxation, for each flow type:
#   LEAPFROG          Γ1 = Γ2 = +π  (CS diffusion):
#     leapfrog_relax, leapfrog_les
#   HEAD-ON COLLISION Γ1 = +π, Γ2 = -π  (GBD diffusion; the bare run blows up):
#     collide_relax, collide_les
#
# Tune by re-running with a different factor, e.g.  RELAX_FACTOR=0.2 ./allrun.sh
#
# Figures: rings_stability.png (max|Γ|) and rings_energy_budget.png (Σ|Γ|, energy,
# enstrophy).  Each case writes stability_metrics.csv (per step), flow_integrals.csv
# and energy_budget.csv.  Existing result directories are deleted before rerun.

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

# Resolution / time-step rationale (ring R=1, core a=0.1, Γ=π, Re=Γ/ν=3000,
# particle core σ_p = 2h):
#   • h=0.030 → 2a/h ≈ 6.7 particles across the core diameter (LES minimum),
#     σ_p = 0.6a, N ≈ 90 k.  h=0.025 is higher fidelity (~1.7× cost).
#   • dt is bound by the vortex-stretching CFL  dt·S_max ≲ 0.2  (S_max ≈ 3 s⁻¹),
#     not diffusion — GBD only needs dt ≤ h²/6ν_eff ≈ 0.087 s at h=0.030.

# Leapfrog (CS): milder strain; dt = 0.020.
LF_DT="${LF_DT:-0.020}"
LF_STEPS="${LF_STEPS:-450}"

# Collision (GBD): dt = 0.06 meets the stretching CFL and is under GBD's bound.
COLLIDE_DT="${COLLIDE_DT:-0.060}"
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-210}}"

# Strength-relaxation tuning knob r ∈ [0,1] and gate.
RELAX_FACTOR="${RELAX_FACTOR:-0.1}"
RELAX_GATE="${RELAX_GATE:-constant}"

# Viscous scheme per flow type (gbd / cs / dvh).
LF_VISCOUS="${LF_VISCOUS:-cs}"
COLLIDE_VISCOUS="${COLLIDE_VISCOUS:-gbd}"

BACKUP_FREQUENCY="${BACKUP_FREQUENCY:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
ENERGY_AUDIT_FREQUENCY="${ENERGY_AUDIT_FREQUENCY:-1}"
DEVICE="${DEVICE:-vulkan}"

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"
echo "Results root: $RUN_ROOT"
echo "Particle spacing: $PARTICLE_SPACING"
echo "Relaxation: factor r=$RELAX_FACTOR, gate=$RELAX_GATE"
echo "Viscous (leapfrog / collide): $LF_VISCOUS / $COLLIDE_VISCOUS"
echo "Backend: $DEVICE"

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
# LEAPFROGGING: Γ1 = Γ2 = +π  (CS diffusion)
# ---------------------------------------------------------------------------

run_case "1/4 leapfrog_relax — Winckelmans/Pedrizzetti relaxation (r=$RELAX_FACTOR)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous "$LF_VISCOUS" \
    --device "$DEVICE" \
    --relaxation --relaxation-factor "$RELAX_FACTOR" --relaxation-gate "$RELAX_GATE" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name leapfrog_relax

run_case "2/4 leapfrog_les — bare LES (reference)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous "$LF_VISCOUS" \
    --device "$DEVICE" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name leapfrog_les

# ---------------------------------------------------------------------------
# HEAD-ON COLLISION: Γ1 = +π, Γ2 = -π  (GBD diffusion — bare run blows up)
# ---------------------------------------------------------------------------

run_case "3/4 collide_relax — Winckelmans/Pedrizzetti relaxation (r=$RELAX_FACTOR)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$COLLIDE_DT" --num-steps "$COLLIDE_STEPS" \
    --viscous "$COLLIDE_VISCOUS" \
    --device "$DEVICE" \
    --relaxation --relaxation-factor "$RELAX_FACTOR" --relaxation-gate "$RELAX_GATE" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name collide_relax

run_case "4/4 collide_les — bare LES (the disease / reference)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" \
    --particle-spacing "$PARTICLE_SPACING" \
    --dt "$COLLIDE_DT" --num-steps "$COLLIDE_STEPS" \
    --viscous "$COLLIDE_VISCOUS" \
    --device "$DEVICE" \
    --backup-frequency "$BACKUP_FREQUENCY" \
    --logging-frequency "$LOGGING_FREQUENCY" \
    --energy-audit-frequency "$ENERGY_AUDIT_FREQUENCY" \
    --name collide_les

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
