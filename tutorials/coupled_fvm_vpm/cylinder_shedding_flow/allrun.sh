#!/usr/bin/env bash
# Run the coupled cylinder-shedding case.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh
mkdir -p solution

python -u cylinder_setup.py 2>&1 | tee solution/cylinder_shedding_flow.stdout.log