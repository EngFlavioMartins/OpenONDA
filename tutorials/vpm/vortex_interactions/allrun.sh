#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

cases=(
    leapfrog_dns leapfrog_les leapfrog_les_stabilized
    collide_dns collide_les collide_les_stabilized
)

if (( $# )); then
    cases=("$@")
fi

for case_name in "${cases[@]}"; do
    if [[ -d "solution/$case_name" ]]; then
        echo "Skipping existing case: $case_name"
        continue
    fi
    echo "===== $case_name ====="
    "$python_bin" -u rings_setup.py --case "$case_name"
done

./allplot.sh png
./allplot.sh pdf
