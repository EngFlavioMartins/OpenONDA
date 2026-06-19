#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh
python stepProfile_setup.py --end-time 1.0 --dt 0.02 --nx 50 --ny 5
./allplot.sh
echo
echo "All runs and plots complete."
