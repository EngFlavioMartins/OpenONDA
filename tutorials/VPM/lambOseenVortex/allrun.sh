#!/usr/bin/env bash
# Lamb-Oseen vortex tutorial — explicit run and plot script.
#
# Each block runs ONE scheme for one test case.  Each invocation is
# a separate Python process so Taichi GPU memory is fully released
# between schemes (prevents OOM on the DVH grid allocation).
#
# No loops, no conditionals, no helper functions.
# Every Python call is fully self-contained and explicit.
#
# All three cases share the same Reynolds number and particle
# resolution to enable a fair comparison of viscous schemes:
#   Gamma = 1.0,  Re = Gamma/nu = 530  =>  nu = 1/530 ~ 1.887e-3
#   a0/b0 = 0.125, b0 = 1.0   =>  a0 = 0.125
#   t0 = a0^2 / (4*nu) ~ 2.071 s   (initial core-radius age)
#   particle spacing h = 0.03375 m  (= 0.27 * a0)
#   rc_ref = √(4 nu t0)   = 0.125 m          (initial core radius)
#   b0     = 1.0 m                           (pair separation)
#   Uc_ref = 1/(2π·0.125) ≃ 1.2732 m s⁻¹
#   ωc_ref = (Γ/(π r_{c}^{2})) = 1/(π·0.125²) ≃ 20.37 s⁻¹

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
