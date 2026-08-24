#!/usr/bin/env bash
# Run the moving- and fixed-plate angle-of-attack sweeps, then make the figures.
# Usage: ./allrun.sh
set -euo pipefail

cd "$(dirname "$0")"
export OPENONDA_COMPUTE_DEVICE="${OPENONDA_COMPUTE_DEVICE:-METAL}"

echo
echo "===== CLEAN ====="
echo
./allclean.sh
mkdir -p solution

echo
echo "===== SIMULATE ====="
echo

run_case() {
    local mode="$1"
    local angle="$2"
    local sign=""
    local magnitude="$angle"
    if (( angle < 0 )); then
        sign="n"
        magnitude=$((-angle))
    fi
    local tag
    printf -v tag 'aoa%s%02d' "$sign" "$magnitude"
    echo
    echo "---- ${mode} plate, angle ${angle} deg ----"
    python -u setup_plate.py --mode "$mode" --angle "$angle" \
        2>&1 | tee "solution/${mode}_${tag}.log"
}

for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    run_case moving "$angle"
done

for angle in -10 -5 -2 0 2 5 8 10 12 15; do
    run_case static "$angle"
done

echo
echo "===== FIGURES ====="
echo
python assets/validate_results.py --pre-plot
./plot_all.sh png
./plot_all.sh pdf

echo
echo "===== VALIDATE ====="
echo
python assets/validate_results.py

echo
echo "===== DONE ====="
echo
