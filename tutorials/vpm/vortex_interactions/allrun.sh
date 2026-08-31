#!/usr/bin/env bash
# Run the four leapfrogging particle-stabilization comparisons.
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh
mkdir -p solution samples figures

run_case() {
    local case_name="$1"
    local exit_code

    echo
    echo "===== ${case_name} ====="
    if "$python_bin" -u setup.py --case "$case_name"; then
        echo "${case_name}: solver completed"
    else
        exit_code=$?
        echo "${case_name}: solver stopped with exit code ${exit_code}; continuing the comparison" >&2
    fi
}

run_case leapfrog_les
run_case leapfrog_les_splitting
run_case leapfrog_les_remeshing
# Run the fully stabilized variant last so earlier experimental failures cannot
# prevent it from receiving the complete requested horizon.
run_case leapfrog_les_splitting_remeshing

./allplot.sh png
"$python_bin" assets/validate_stabilization_suite.py
