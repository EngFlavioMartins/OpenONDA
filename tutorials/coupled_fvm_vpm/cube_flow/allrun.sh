#!/usr/bin/env bash
# Run and validate the configured coupled cube-flow case.
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: ./allrun.sh" >&2
    exit 2
fi

readonly reference_end_time="0.10"
readonly acceptance_limit="0.05"
readonly python_bin="${OPENONDA_PYTHON:-python}"
readonly case_directory="$PWD"
readonly reference_cache_input="${OPENONDA_CUBE_REFERENCE:-${TMPDIR:-/tmp}/openonda_cube_flow_fvm_reference_t010}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[FAIL] Python executable not found: $python_bin" >&2
    echo "Activate the OpenONDA environment or set OPENONDA_PYTHON." >&2
    exit 1
fi
reference_cache="$("$python_bin" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$reference_cache_input")"
readonly reference_cache

for required_file in \
    cube_flow_setup.py \
    allclean.sh \
    assets/check_run.py \
    reference_flow/assets/run_trial.py; do
    if [[ ! -f "$required_file" ]]; then
        echo "[FAIL] Missing required case file: $required_file" >&2
        exit 1
    fi
done

if ! preflight_output=$("$python_bin" -c \
    'import openonda.coupler, openonda.fvm, openonda.vpm, scipy, taichi; import cube_flow_setup as c; assert c.END_TIME == 20.0; assert c.FVM_TIME_STEP_SIZE == 0.005; assert c.VPM_TIME_STEP_SIZE == 0.010; assert c.VPM_PARTICLE_SPACING == 0.03125; assert c.VPM_VISCOUS_SCHEME == "GBD"; assert c.VPM_PANEL_SOLVER.linear_solver_name == "SCIPY"; assert c.VPM_PANEL_SOLVER.coupling_scope == "vpm_boundary_condition"; assert c.COUPLER_SETUP.transfer_method == "common_lattice"; assert c.COUPLER_SETUP.eta_blend_width == 3 * c.VPM_PARTICLE_SPACING' \
    2>&1); then
    printf '%s\n' "$preflight_output" >&2
    echo "[FAIL] Cube-flow Python/configuration preflight failed." >&2
    exit 1
fi

reference_is_ready() {
    "$python_bin" -c \
        'import csv, pathlib, sys; root = pathlib.Path(sys.argv[1]) / "samples"; names = ("forces_history.csv", "centreline.csv", "offaxis_y075.csv"); assert all((root / name).is_file() for name in names); rows = list(csv.DictReader((root / names[0]).open())); assert rows and float(rows[-1]["time"]) >= 0.1 - 1e-12' \
        "$reference_cache" >/dev/null 2>&1
}

print_configuration() {
    printf '%s\n' \
        "===== CUBE-FLOW RUN =====" \
        "python=$(command -v "$python_bin")" \
        "end_time=20.0" \
        "fvm_dt=0.005" \
        "vpm_dt=0.010" \
        "viscous_scheme=GBD" \
        "panel_solver=SCIPY" \
        "panel_scope=vpm_boundary_condition" \
        "transfer=common lattice, h=0.03125, blend_width=3h" \
        "reference=${reference_cache}" \
        "acceptance_limit=${acceptance_limit}"
}

print_configuration
if reference_is_ready; then
    echo "reference_status=ready"
else
    echo "reference_status=will be generated (fully meshed FVM to t=${reference_end_time})"
fi

prepare_reference() {
    if reference_is_ready; then
        echo "===== REFERENCE: REUSE CURRENT CACHE ====="
        return
    fi

    if [[ "$reference_cache" == "/" || "$reference_cache" == "$case_directory" || "$reference_cache" == "$case_directory/"* ]]; then
        echo "[FAIL] Refusing to replace a reference cache inside the cube source case: $reference_cache" >&2
        echo "Choose an external OPENONDA_CUBE_REFERENCE path." >&2
        exit 1
    fi

    if [[ -e "$reference_cache" ]]; then
        stale_reference="${reference_cache}.stale.$(date -u +'%Y%m%dT%H%M%SZ').$$"
        echo "Moving the previous reference cache to: $stale_reference"
        mv "$reference_cache" "$stale_reference"
    fi
    mkdir -p "$reference_cache"

    echo "===== REFERENCE: FRESH FULLY-MESHED FVM TO t=${reference_end_time} ====="
    "$python_bin" -u reference_flow/assets/run_trial.py \
        --end-time "$reference_end_time" \
        --output-directory "$reference_cache" \
        2>&1 | tee "$reference_cache/reference_runner.log"
    if ! reference_is_ready; then
        echo "[FAIL] The reference run did not produce complete t=0.10 samples." >&2
        exit 1
    fi
}

prepare_reference

echo "===== CLEAN COUPLED CASE ====="
./allclean.sh

echo "===== COUPLED CUBE RUN ====="
mkdir -p solution
"$python_bin" -u cube_flow_setup.py 2>&1 | tee solution/cube_flow.log

echo "===== VALIDATE SOLVER, DRAG, AND VELOCITY PROFILES ====="
"$python_bin" assets/check_run.py \
    --case-directory "$case_directory" \
    --reference-directory "$reference_cache" \
    --acceptance-limit "$acceptance_limit"

echo "[OK] Cube-flow run and validation completed."
