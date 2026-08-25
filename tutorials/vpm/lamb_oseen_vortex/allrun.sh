#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    echo "Usage: $0 [--dry-run]"
    echo "  --dry-run  Validate dependencies and print the pinned campaign without cleaning or running."
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

# Accepted full-length resource profile. These values are deliberately pinned
# here so a stale shell environment cannot silently change a production run.
readonly device="METAL"
readonly spacing_ratio="0.45"
readonly column_spacing_ratio="0.45"
readonly strength_cutoff="0.01"
readonly dvh_rd_ratio="4"
readonly dvh_padding="5"
readonly gbd_padding="5"
readonly dvh_threshold="1e-4"
readonly gbd_threshold="5e-5"
readonly max_nodes="300000"
readonly direct_integral_limit="300000"

python_bin="${OPENONDA_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[FAIL] Python executable not found: $python_bin" >&2
    echo "Activate the OpenONDA environment or set OPENONDA_PYTHON." >&2
    exit 1
fi

for required_file in \
    lamb_oseen_setup.py \
    assets/validate_results.py \
    allclean.sh \
    allplot.sh; do
    if [[ ! -f "$required_file" ]]; then
        echo "[FAIL] Missing required campaign file: $required_file" >&2
        exit 1
    fi
done

if ! "$python_bin" -c 'import openonda.vpm; import taichi' >/dev/null; then
    echo "[FAIL] The selected Python cannot import OpenONDA VPM and Taichi." >&2
    exit 1
fi

# Make allplot.sh resolve the same interpreter selected above.
python_path=$(command -v "$python_bin")
export PATH="$(dirname "$python_path"):$PATH"

case_names=(
    vortex_cs vortex_rwm vortex_dvh vortex_gbd
    dipole_cs dipole_rwm dipole_dvh dipole_gbd
    merging_cs merging_rwm merging_dvh merging_gbd
)

print_configuration() {
    printf '%s\n' \
        "===== LAMB--OSEEN PRODUCTION CAMPAIGN =====" \
        "python=${python_bin}" \
        "backend=${device}" \
        "column_length=50*a0 (6.25 m; fixed in lamb_oseen_setup.py)" \
        "spacing_ratio=${spacing_ratio}" \
        "column_spacing_ratio=${column_spacing_ratio}" \
        "field_spacing_ratio=0.15" \
        "strength_cutoff=${strength_cutoff}" \
        "dvh_rd_ratio=${dvh_rd_ratio}" \
        "dvh_threshold=${dvh_threshold}" \
        "gbd_threshold=${gbd_threshold}" \
        "dvh_padding_cells=${dvh_padding}" \
        "gbd_padding_cells=${gbd_padding}" \
        "max_regeneration_nodes=${max_nodes}" \
        "direct_integral_limit=${direct_integral_limit}" \
        "time_step=0.291/9 s" \
        "end_time=103*0.291 s" \
        "cases=${case_names[*]}"
}

print_configuration
if [[ "$dry_run" == true ]]; then
    echo "[OK] Dry run passed; no files were removed and no simulation was started."
    exit 0
fi

echo "===== METAL PREFLIGHT ====="
"$python_bin" -c \
    'import taichi as ti; ti.init(arch=ti.metal, default_fp=ti.f32, offline_cache=False); probe = ti.field(dtype=ti.i32, shape=()); probe[None] = 1; assert probe[None] == 1; ti.sync(); ti.reset(); print("[OK] Taichi Metal field allocation passed.")'

# Pin all environment-controlled solver values to the accepted profile.
export OPENONDA_COMPUTE_DEVICE="$device"
export OPENONDA_VPM_DIRECT_INTEGRAL_LIMIT="$direct_integral_limit"
export OPENONDA_LAMB_DVH_RD_RATIO="$dvh_rd_ratio"
export OPENONDA_LAMB_DVH_THRESHOLD="$dvh_threshold"
export OPENONDA_LAMB_GBD_THRESHOLD="$gbd_threshold"
export OPENONDA_LAMB_DVH_MAX_NODES="$max_nodes"
export OPENONDA_LAMB_GBD_MAX_NODES="$max_nodes"
export OPENONDA_PYTHON="$python_bin"

echo "===== CLEAN START ====="
echo "Removing existing solution/, samples/, and figures/ after successful preflight."
./allclean.sh
mkdir -p solution

{
    print_configuration
    printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'python_version=%s\n' "$($python_bin --version 2>&1)"
    if git_sha=$(git rev-parse HEAD 2>/dev/null); then
        printf 'git_sha=%s\n' "$git_sha"
    fi
} | tee solution/campaign_configuration.txt

common_args=(
    --spacing-ratio "$spacing_ratio"
    --column-spacing-ratio "$column_spacing_ratio"
    --strength-cutoff "$strength_cutoff"
    --dvh-rd-ratio "$dvh_rd_ratio"
    --dvh-padding "$dvh_padding"
    --gbd-padding "$gbd_padding"
    --dvh-threshold "$dvh_threshold"
    --gbd-threshold "$gbd_threshold"
    --dvh-max-nodes "$max_nodes"
    --gbd-max-nodes "$max_nodes"
)

run_case() {
    local case_name="$1"
    shift
    echo "===== ${case_name} (${device}) ====="
    "$python_bin" -u lamb_oseen_setup.py "$@" "${common_args[@]}" \
        --case-name "$case_name" --compute-device "$device" \
        2>&1 | tee "solution/${case_name}.log"
}

run_case vortex_cs --circulation1 +1 --circulation2 0 --viscous-scheme CS
run_case vortex_rwm --circulation1 +1 --circulation2 0 --viscous-scheme RWM
run_case vortex_dvh --circulation1 +1 --circulation2 0 --viscous-scheme DVH
run_case vortex_gbd --circulation1 +1 --circulation2 0 --viscous-scheme GBD
run_case dipole_cs --circulation1 +1 --circulation2 -1 --viscous-scheme CS
run_case dipole_rwm --circulation1 +1 --circulation2 -1 --viscous-scheme RWM
run_case dipole_dvh --circulation1 +1 --circulation2 -1 --viscous-scheme DVH
run_case dipole_gbd --circulation1 +1 --circulation2 -1 --viscous-scheme GBD
run_case merging_cs --circulation1 +1 --circulation2 +1 --viscous-scheme CS
run_case merging_rwm --circulation1 +1 --circulation2 +1 --viscous-scheme RWM
run_case merging_dvh --circulation1 +1 --circulation2 +1 --viscous-scheme DVH
run_case merging_gbd --circulation1 +1 --circulation2 +1 --viscous-scheme GBD

"$python_bin" assets/validate_results.py --pre-plot
./allplot.sh png
./allplot.sh pdf
"$python_bin" assets/validate_results.py

echo "[OK] All 12 cases, both figure formats, and strict validation completed."
