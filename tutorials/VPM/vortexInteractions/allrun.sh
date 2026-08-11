#!/bin/sh
set -eu

cd "$(dirname "$0")"

CASES="leapfrog_dns leapfrog_les leapfrog_les_stabilized collide_dns collide_les collide_les_stabilized"
for case_name in ${*:-$CASES}; do
    python -u rings_setup.py "$case_name"
done

python assets/check_run.py "$@"
