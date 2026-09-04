#!/usr/bin/env bash
# Run and validate the configured coupled cube-flow case.
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: ./allrun.sh" >&2
    exit 2
fi

readonly reference_horizon="20.00"
readonly acceptance_horizon="2.00"
readonly acceptance_limit="0.07"
readonly python_bin="${OPENONDA_PYTHON:-python}"
readonly case_directory="$PWD"
readonly reference_input="${OPENONDA_CUBE_REFERENCE:-$case_directory/reference_flow}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[FAIL] Python executable not found: $python_bin" >&2
    echo "Activate the OpenONDA environment or set OPENONDA_PYTHON." >&2
    exit 1
fi
reference_directory="$("$python_bin" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$reference_input")"
readonly reference_directory

for required_file in \
    setup.py \
    allclean.sh \
    assets/check_run.py; do
    if [[ ! -f "$required_file" ]]; then
        echo "[FAIL] Missing required case file: $required_file" >&2
        exit 1
    fi
done

if ! preflight_output=$("$python_bin" -c \
    'import openonda.coupler, openonda.fvm, openonda.vpm, scipy, taichi; import setup as c; assert c.END_TIME == 20.0; assert c.FVM_TIME_STEP_SIZE == 0.010; assert c.VPM_TIME_STEP_SIZE == 0.050; assert c.VPM_PARTICLE_SPACING == 0.03125; assert c.VPM_VISCOUS_SCHEME == "GBD"; assert c.VPM_PANEL_SOLVER.linear_solver_name == "SCIPY"; assert c.VPM_PANEL_SOLVER.coupling_scope == "vpm_boundary_condition"; assert c.COUPLER_SETUP.transfer_method == "buffered_m4_renewal"; assert c.COUPLER_SETUP.eta_blend_width == 6 * c.VPM_PARTICLE_SPACING; assert c.VPM_CASE.numerics.write_precision == "f32"; assert c.VPM_CASE.backup.interval_steps == 0' \
    2>&1); then
    printf '%s\n' "$preflight_output" >&2
    echo "[FAIL] Cube-flow Python/configuration preflight failed." >&2
    exit 1
fi

reference_is_ready() {
    "$python_bin" -c \
        'import csv, pathlib, sys; root = pathlib.Path(sys.argv[1]) / "samples"; horizon = float(sys.argv[2]); names = ("forces_history.csv", "centreline.csv", "offaxis_y075.csv"); assert all((root / name).is_file() for name in names); tables = [list(csv.DictReader((root / name).open())) for name in names]; assert all(rows and max(float(row["time"]) for row in rows) >= horizon - 1e-12 for rows in tables)' \
        "$reference_directory" "$reference_horizon" >/dev/null 2>&1
}

print_configuration() {
    printf '%s\n' \
        "===== CUBE-FLOW RUN =====" \
        "python=$(command -v "$python_bin")" \
        "end_time=20.0" \
        "fvm_dt=0.010" \
        "vpm_dt=0.050" \
        "viscous_scheme=GBD" \
        "panel_solver=SCIPY" \
        "panel_scope=vpm_boundary_condition" \
        "transfer=buffered M4' whole-belt renewal, h=0.03125, blend_width=6h" \
        "fvm_consistency=resolved-scale outer 0.25m buffer" \
        "particle_history=solution/vpm_STEP.{h5,xdmf}, post-renewal f32" \
        "reference=${reference_directory}" \
        "reference_horizon=${reference_horizon}" \
        "acceptance_horizon=${acceptance_horizon}" \
        "acceptance_limit=${acceptance_limit}"
}

print_configuration
if ! reference_is_ready; then
    if [[ "$reference_directory" == "$case_directory/reference_flow" ]] && \
       [[ -x "$reference_directory/allrun.sh" ]]; then
        echo "reference_status=missing; generating the bundled reference case"
        (cd "$reference_directory" && ./allrun.sh)
    fi
fi
if ! reference_is_ready; then
    echo "[FAIL] The selected reference must contain forces and both profiles through t=${reference_horizon}: ${reference_directory}" >&2
    exit 1
fi
echo "reference_status=full-horizon archive ready"

echo "===== CLEAN COUPLED CASE ====="
./allclean.sh

echo "===== COUPLED CUBE RUN ====="
mkdir -p solution
"$python_bin" -u setup.py 2>&1 | tee solution/cube_flow.log

echo "===== VALIDATE SOLVER, DRAG, AND VELOCITY PROFILES ====="
"$python_bin" assets/check_run.py \
    --case-directory "$case_directory" \
    --reference-directory "$reference_directory" \
    --acceptance-limit "$acceptance_limit" \
    --acceptance-horizon "$acceptance_horizon"

echo "[OK] Cube-flow run and validation completed."
