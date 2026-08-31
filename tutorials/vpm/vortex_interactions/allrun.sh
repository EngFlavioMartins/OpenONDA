#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

cases=(
    leapfrog_les leapfrog_les_rvpm leapfrog_les_rvpm_sfs
    collide_les collide_les_rvpm_sfs
)

for case_name in "${cases[@]}"; do
    if [[ -d "solution/$case_name" ]]; then
        echo "Skipping existing case: $case_name"
        continue
    fi
    "$python_bin" -u rings_setup.py --case "$case_name"
done

"$python_bin" assets/check_leapfrogging.py
./allplot.sh png
./allplot.sh pdf
