#!/usr/bin/env bash
# Run the staged vortex-ring interaction experiment.
#
# Stage 1 always runs four fixed-particle transposed baselines:
#   leapfrog/collide x DNS/LES.
# Stage 2 reruns only a family whose plain LES failed before the requested
# turbulent-transition horizon. That retry adds overshoot-gated filament
# splitting; it does not remesh or continuously relax the full cloud.
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    echo "Usage: $0 [--dry-run]"
}

dry_run=false
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

readonly python_bin="${OPENONDA_PYTHON:-python}"
readonly num_steps="6000"
readonly device="METAL"
readonly les_cs="0.20"
readonly particle_spacing="0.035"
readonly widnall_amplitude="0.05"
readonly widnall_modes="24"

baseline_cases=(
    leapfrog_dns
    leapfrog_les
    collide_dns
    collide_les
)

print_configuration() {
    printf '%s\n' \
        "===== VORTEX-INTERACTIONS STAGED CAMPAIGN =====" \
        "backend=${device}" \
        "baseline_cases=${baseline_cases[*]}" \
        "stretching=transposed" \
        "les_smagorinsky_coefficient=${les_cs}" \
        "particle_spacing=${particle_spacing}" \
        "widnall_amplitude=${widnall_amplitude}" \
        "widnall_modes=1..${widnall_modes}" \
        "requested_steps=${num_steps}" \
        "requested_nondimensional_time=147" \
        "conditional_stabilization=overshoot-gated filament splitting" \
        "remeshing=disabled" \
        "strength_reorientation=disabled"
}

print_configuration
if [[ "$dry_run" == true ]]; then
    "$python_bin" -c 'import openonda.vpm; import taichi' >/dev/null
    echo "[OK] Dry run passed; no files were removed and no simulations were started."
    exit 0
fi

export OPENONDA_COMPUTE_DEVICE="$device"
export OPENONDA_INTERACTIONS_NUM_STEPS="$num_steps"
export OPENONDA_INTERACTIONS_VELOCITY_METHOD="treecode"

echo
echo "===== METAL PREFLIGHT ====="
"$python_bin" -c \
    'import taichi as ti; ti.init(arch=ti.metal, default_fp=ti.f32, offline_cache=False); probe = ti.field(dtype=ti.i32, shape=()); probe[None] = 1; assert probe[None] == 1; ti.sync(); ti.reset(); print("[OK] Taichi Metal field allocation passed.")'

echo
echo "===== CLEAN START ====="
./allclean.sh --all

run_case() {
    local case_name="$1"
    echo
    echo "===== ${case_name} ====="
    "$python_bin" -u rings_setup.py --case "$case_name"
}

case_status() {
    local case_name="$1"
    "$python_bin" -c \
        'import json, pathlib, sys; path = pathlib.Path("solution") / sys.argv[1] / "run_manifest.json"; print(json.loads(path.read_text()).get("status", "missing") if path.is_file() else "missing")' \
        "$case_name"
}

ran_cases=()
for case_name in "${baseline_cases[@]}"; do
    run_case "$case_name"
    ran_cases+=("$case_name")
done

echo
echo "===== CONDITIONAL STABILIZATION GATE ====="
for family in leapfrog collide; do
    plain_case="${family}_les"
    status="$(case_status "$plain_case")"
    if [[ "$status" == "completed" ]]; then
        echo "[SKIP] ${family}: plain calibrated LES reached the full transition horizon."
        continue
    fi
    stabilized_case="${family}_les_stabilized"
    echo "[RUN] ${family}: plain LES status=${status}; enabling overshoot-gated splitting."
    run_case "$stabilized_case"
    ran_cases+=("$stabilized_case")
done

echo
echo "===== VALIDATE ====="
validation_status=0
"$python_bin" assets/check_run.py "${ran_cases[@]}" || validation_status=$?

echo
echo "===== FIGURES ====="
./allplot.sh png --allow-partial
./allplot.sh pdf --allow-partial

echo
if [[ "$validation_status" -eq 0 ]]; then
    echo "[OK] Staged campaign, physics checks, and figures completed."
else
    echo "[WARN] Simulations and figures completed, but the physics gate reported failures."
fi
exit "$validation_status"
