#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

cases=(
    leapfrog_les leapfrog_les_splitting leapfrog_les_realignment
    collide_les collide_les_realignment
)

for case_name in "${cases[@]}"; do
    if [[ -d "solution/$case_name" ]]; then
        echo "Skipping existing case: $case_name"
        continue
    fi
    "$python_bin" -u rings_setup.py --case "$case_name"
done

./allplot.sh png
./allplot.sh pdf
