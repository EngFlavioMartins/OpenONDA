#!/usr/bin/env bash
# Vortex-ring interactions — stabilization LADDER demonstration.
#
# Objective
# ---------
# Show that adding the LES model, and subsequently the strength-relaxation
# (ISR) stabilizer, progressively increases stability while keeping the
# solution physically grounded:
#
#     DNS        →  may fail (blow up) first
#     +LES       →  fails later
#     +LES+ISR   →  survives longest
#
# Lower rungs ARE allowed to fail — what matters is the failure ordering and
# that the surviving rungs track the physics (LBM leapfrog trajectory data in
# assets/, conserved invariants, sensible energy decay).
#
# Numerical ground rules (post-mortem driven — see docs/vpm_stabilization_audit.md):
#   * dt = Δt_d ≈ 0.103 s (the DVH-pinned step — the diffusion operator
#     fires once per step): advective CFL ≈ 4.  Refine h (which shrinks
#     Δt_d = β·R_d²/(4nu) ∝ h²) or switch to GBD/CS for a smaller CFL.
#   * DVH regen threshold in BUDGET mode (≤2e-4 of Σ|Γ| lost per firing) with
#     a 250k node cap: the old absolute threshold (3e-5) destroyed ~1.2% of
#     the circulation PER FIRING (total evaporation by step ~450) — the rings
#     merged and dissolved unphysically in every case.
#   * All cases share the GRADU stretching operator so the only difference
#     between rungs is the stabilization layer.
#
# Case matrix:
#   LEAPFROG (physics fidelity; rungs likely all survive — compare vs LBM):
#     1. leapfrog_dns      2. leapfrog_les      3. leapfrog_les_isr
#     4. leapfrog_les_pedr (|Γ|-preserving Pedrizzetti variant)
#   COLLISION (stability ladder — strain peaks at impact):
#     5. collide_dns       6. collide_les       7. collide_les_isr
#
# LES+relaxation cases run on Vulkan (Taichi 1.7.4 CUDA-backend bug with this
# combination — see rings_setup --device help).
#
# A blow-up is detected by rings_setup.py (max|Γ| > 50× initial) — the run
# stops, saves a pre_blowup state, and still counts as PASS; survival time is
# read from the logs / figures/compare_summary.csv (last_step column).
#
# Usage:
#   ./allrun.sh                  # full ladder (120 steps ≈ 12.4 s physical)
#   N_STEPS=100 ./allrun.sh      # quick smoke-test
#
# Requires the OpenONDA conda environment:  conda activate OpenONDA

set -uo pipefail   # -e removed: individual case failures must not stop the run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$(conda run -n OpenONDA which python 2>/dev/null \
          || command -v python3 \
          || command -v python)"

GAMMA_PI="3.14159265358979"
DT="0.103"          # Δt_d — DVH fires once per step (diffusion active every step)

# 120 × 0.103 ≈ 12.4 s physical — covers the LBM range x/R0 ≲ 10;
# collision impact at t ≈ 4.2 s.
STEPS="${N_STEPS:-120}"

_RESULTS=()

run_case() {
    local label="$1"; shift
    echo ""
    echo "========================================================================"
    echo "${label}"
    echo "========================================================================"
    local rc=0
    $PYTHON rings_setup.py "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "→ ${label} complete."
        _RESULTS+=("PASS — ${label}")
    else
        echo "*** ${label} exited with code ${rc} — continuing ***" >&2
        _RESULTS+=("FAIL (exit ${rc}) — ${label}")
    fi
}

if [[ "${CLEAN:-1}" == "1" ]]; then
    echo "Cleaning previous results…"
    ./allclean.sh
fi

# ═════════════════════════════════════════════════════════════════════════════
# LEAPFROGGING  (Γ₁ = Γ₂ = +π) — physics fidelity vs LBM reference
# ═════════════════════════════════════════════════════════════════════════════

run_case "1/7 leapfrog_dns — DNS baseline" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode dns \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu \
    --name leapfrog_dns

run_case "2/7 leapfrog_les — +LES" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu \
    --name leapfrog_les

run_case "3/7 leapfrog_les_isr — +ISR (conservative ADM blend, C=1.5)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu --relaxation blend --isr-C 1.5 --deconv 1 \
    --device vulkan \
    --name leapfrog_les_isr

run_case "4/7 leapfrog_les_pedr — +Pedrizzetti variant (|Γ|-preserving)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu --relaxation pedrizzetti --isr-C 1.5 --deconv 1 \
    --device vulkan \
    --name leapfrog_les_pedr

# ═════════════════════════════════════════════════════════════════════════════
# HEAD-ON COLLISION  (Γ₁ = +π, Γ₂ = −π) — the stability ladder
# ═════════════════════════════════════════════════════════════════════════════

run_case "5/7 collide_dns — DNS baseline (expected to fail first)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode dns \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu \
    --name collide_dns

run_case "6/7 collide_les — +LES (expected to fail later)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu \
    --name collide_les

run_case "7/7 collide_les_isr — +ISR (expected to survive longest)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$STEPS" \
    --stretching gradu --relaxation blend --isr-C 1.5 --deconv 1 \
    --device vulkan \
    --name collide_les_isr

# ── summary ───────────────────────────────────────────────────────────────────

echo ""
echo "========================================================================"
echo "Run summary"
echo "========================================================================"
_n_pass=0; _n_fail=0
for entry in "${_RESULTS[@]}"; do
    echo "  ${entry}"
    if [[ "${entry}" == PASS* ]]; then
        (( _n_pass++ ))
    else
        (( _n_fail++ ))
    fi
done
echo ""
echo "  ${_n_pass} passed, ${_n_fail} failed"
echo "  (blow-ups are expected on the lower collision rungs — survival times"
echo "   are in figures/compare_summary.csv after allplot.sh)"
echo ""

if [[ -x ./allplot.sh ]]; then
    echo "Generating comparison figures…"
    ./allplot.sh || true
fi

[[ $_n_fail -eq 0 ]]
