#!/usr/bin/env bash
# Vortex-ring interactions — stabilization LADDER demonstration.
#
# Objective
# ---------
# Compare the existing DNS/LES/relaxation ladder with a finer leapfrogging case
# that resolves the coupled advection--stretching update in complete microsteps:
#
#     DNS        →  may fail (blow up) first
#     +LES       →  fails later
#     +LES+ISR   →  strength-filtered legacy stabilization
#     fine       →  deformation-limited full-state temporal refinement
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
#   LONG LEAPFROG CHALLENGE:
#     8. leapfrog_fine (h=.020, dt=.010, adaptive complete-physics substeps)
#
# LES+relaxation cases run on Vulkan (Taichi 1.7.4 CUDA-backend bug with this
# combination — see rings_setup --device help).
#
# A blow-up is detected by rings_setup.py (max|Γ| > 50× initial) — the run
# stops, saves a pre_blowup state, and still counts as PASS; survival time is
# read from the logs / figures/compare_summary.csv (last_step column).
#
# Usage:
#   ./allrun.sh                         # full ladder
# CASE_SET=fine FINE_STEPS=10 ./allrun.sh
#
# Results are never overwritten. Each case writes to solution/<name>/ and
# rings_setup.py refuses to overwrite a non-empty result directory.
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
COLLIDE_STEPS="${COLLIDE_STEPS:-${N_STEPS:-120}}"

# Per-step stability_metrics.csv stays enabled independently of these cadences.
# Sparse full-state backups keep trajectory reconstruction possible without the
# multi-gigabyte output produced by the historical every-2-step setting.
LEGACY_BACKUP_FREQUENCY="${LEGACY_BACKUP_FREQUENCY:-20}"
LEGACY_LOGGING_FREQUENCY="${LEGACY_LOGGING_FREQUENCY:-10}"

# --- Leapfrog rungs: stable, well-resolved settings -------------------------
LF_DT="0.02"
LF_STEPS="${LF_STEPS:-600}"
FINE_STEPS="${FINE_STEPS:-2000}"
RUN_ROOT="${RUN_ROOT:-solution}"
FIGURES_ROOT="${FIGURES_ROOT:-figures}"
CASE_SET="${CASE_SET:-all}"  # all | legacy | fine

if [[ "$CASE_SET" != "all" && "$CASE_SET" != "legacy" && "$CASE_SET" != "fine" ]]; then
    echo "CASE_SET must be one of: all, legacy, fine (got: $CASE_SET)" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT" "$FIGURES_ROOT"
echo "Results root: $RUN_ROOT"

_RESULTS=()

run_case() {
    local label="$1"; shift
    echo ""
    echo "========================================================================"
    echo "${label}"
    echo "========================================================================"
    local rc=0
    $PYTHON rings_setup.py --output-root "$RUN_ROOT" "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "→ ${label} complete."
        _RESULTS+=("PASS — ${label}")
    else
        echo "*** ${label} exited with code ${rc} — continuing ***" >&2
        _RESULTS+=("FAIL (exit ${rc}) — ${label}")
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# LEAPFROGGING  (Γ₁ = Γ₂ = +π) — physics fidelity vs LBM reference
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$CASE_SET" != "fine" ]]; then
run_case "1/8 leapfrog_dns — DNS baseline" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode dns \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs --stretching rvpm \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name leapfrog_dns

run_case "2/8 leapfrog_les — +LES" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs --stretching rvpm \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name leapfrog_les

run_case "3/8 leapfrog_les_isr — +ISR (conservative ADM blend, C=1.5)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs --stretching rvpm --relaxation blend --relaxation-rate 1.5 --deconv 1 \
    --device vulkan \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name leapfrog_les_isr

run_case "4/8 leapfrog_les_pedr — +Pedrizzetti variant (|Γ|-preserving)" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --dt "$LF_DT" --num-steps "$LF_STEPS" \
    --viscous cs --stretching rvpm --relaxation pedrizzetti --relaxation-rate 1.5 --deconv 1 \
    --device vulkan \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name leapfrog_les_pedr

# ═════════════════════════════════════════════════════════════════════════════
# HEAD-ON COLLISION  (Γ₁ = +π, Γ₂ = −π) — the stability ladder
# ═════════════════════════════════════════════════════════════════════════════

run_case "5/8 collide_dns — DNS baseline (expected to fail first)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode dns \
    --dt "$DT" --num-steps "$COLLIDE_STEPS" \
    --stretching gradu \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name collide_dns

run_case "6/8 collide_les — +LES (expected to fail later)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$COLLIDE_STEPS" \
    --stretching gradu \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name collide_les

run_case "7/8 collide_les_isr — +ISR (expected to survive longest)" \
    --gamma1 "$GAMMA_PI" --gamma2 "-$GAMMA_PI" --mode les \
    --dt "$DT" --num-steps "$COLLIDE_STEPS" \
    --stretching gradu --relaxation blend --relaxation-rate 1.5 --deconv 1 \
    --device vulkan \
    --backup-frequency "$LEGACY_BACKUP_FREQUENCY" \
    --logging-frequency "$LEGACY_LOGGING_FREQUENCY" \
    --name collide_les_isr
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FINE LEAPFROG STABILITY CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$CASE_SET" != "legacy" ]]; then
run_case "8/8 leapfrog_fine — fine, full-state deformation subcycling" \
    --gamma1 "$GAMMA_PI" --gamma2 "$GAMMA_PI" --mode les \
    --particle-spacing 0.020 --dt 0.010 --num-steps "$FINE_STEPS" \
    --viscous cs --stretching rvpm --relaxation none \
    --physics-substeps 1 --deformation-cfl 0.15 --max-physics-substeps 64 \
    --backup-frequency 20 --logging-frequency 5 --max-particles 750000 \
    --name leapfrog_fine
fi

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
    ./allplot.sh --solution-dir "$RUN_ROOT" --figures-dir "$FIGURES_ROOT" || true
fi

[[ $_n_fail -eq 0 ]]
