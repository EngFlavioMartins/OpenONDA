#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
python airfoilFlow_setup.py --end-time 30.0 --dt 0.05 --Re 1000 --angle 23
./allplot.sh
echo
echo "All runs and plots complete."
