#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
mkdir -p solution

python taylorGreen_setup.py --n 24 --nu 0.1 --dt 0.005 --end-time 0.05 \
    2>&1 | tee solution/taylorGreen.log

./allplot.sh
echo
echo "All runs and plots complete."
