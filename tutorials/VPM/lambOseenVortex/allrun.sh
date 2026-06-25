#!/usr/bin/env bash
# Lamb-Oseen vortex — viscous-scheme benchmark (CS, RWM, DVH, GBD).
# Runs three cases (single vortex, dipole, co-rotating merger) at Re=530
# to compare diffusion accuracy against the C&W 2003 reference solution.

set -euo pipefail

# Resolve Python interpreter: prefer the OpenONDA conda env, fall back to legacy
# then to whatever python3/python is on PATH.
PYTHON="${OPENONDA_PYTHON:-$(conda run -n OpenONDA which python 2>/dev/null \
       || command -v python3 \
       || command -v python)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clean up old data and figures:
./allclean.sh

# ============================================================
# TEST CASE 1: Single Lamb-Oseen vortex diffusion
# ============================================================
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes cs \
    --re 530 --dt 0.123 --total-time 20.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes rwm \
    --re 530 --dt 0.123 --total-time 20.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes dvh \
    --re 530 --dt 0.123 --total-time 20.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes gbd \
    --re 530 --dt 0.123 --total-time 20.0 \
    --solution-dir ./solution --clean

sleep 15

echo 'DONE WITH SINGLE VORTEX'

# ============================================================
# TEST CASE 2: Vortex dipole (counter-rotating pair)
# ============================================================
# NOTE: the dipole runs 2x longer than the single-vortex case (40 s vs 20 s) so
# the counter-rotating pair convects far enough to study its mutual induction.
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes cs \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes rwm \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes dvh \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes gbd \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean


echo 'DONE WITH DIPOLE'

"$PYTHON" assets/validate_solution.py --solution-dir ./solution \
    --case dipole --expected-time 40.0

sleep 15

# ============================================================
# TEST CASE 3: Co-rotating vortex merger vortex merger
# ============================================================
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes cs \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes rwm \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes dvh \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes gbd \
    --re 530 --dt 0.123 --total-time 40.0 \
    --solution-dir ./solution --clean

./allplot.sh

echo
echo "All runs and plots complete."
