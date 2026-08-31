#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"

./allclean.sh
mkdir -p solution

# Moving plate (body frame, smooth ramp to avoid impulsive-start transient)
echo "===== moving ====="
"$python_bin" -u setup.py --mode moving --angle -10
"$python_bin" -u setup.py --mode moving --angle -5
"$python_bin" -u setup.py --mode moving --angle -2
"$python_bin" -u setup.py --mode moving --angle 0
"$python_bin" -u setup.py --mode moving --angle 2
"$python_bin" -u setup.py --mode moving --angle 5
"$python_bin" -u setup.py --mode moving --angle 8
"$python_bin" -u setup.py --mode moving --angle 10
"$python_bin" -u setup.py --mode moving --angle 12
"$python_bin" -u setup.py --mode moving --angle 15

# Static plate (wind frame, fixed angle of attack)
echo "===== static ====="
"$python_bin" -u setup.py --mode static --angle -10
"$python_bin" -u setup.py --mode static --angle -5
"$python_bin" -u setup.py --mode static --angle -2
"$python_bin" -u setup.py --mode static --angle 0
"$python_bin" -u setup.py --mode static --angle 2
"$python_bin" -u setup.py --mode static --angle 5
"$python_bin" -u setup.py --mode static --angle 8
"$python_bin" -u setup.py --mode static --angle 10
"$python_bin" -u setup.py --mode static --angle 12
"$python_bin" -u setup.py --mode static --angle 15

# Post-processing: validation and figures (png, pdf)
"$python_bin" assets/validate_results.py --pre-plot
./allplot.sh png
./allplot.sh pdf
"$python_bin" assets/validate_results.py
