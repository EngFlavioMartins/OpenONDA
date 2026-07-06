#!/usr/bin/env bash
# Lamb-Oseen vortex — viscous-scheme benchmark (CS, RWM, DVH, GBD).
# Runs three cases (single vortex, dipole, co-rotating merger) at Re=530
# to compare diffusion accuracy against the C&W 2003 reference solution.
#
# ── Buckingham-Pi normalisation ──────────────────────────────────────────────
# Dimensional parameters:
#   $\Gamma$       = 1.0 m^2/s          circulation per vortex
#   $\nu$          = 1/530 m^2/s        kinematic viscosity
#   $b_0$          = 1.0 m              centre-to-centre separation
#   $r_{c,0}$     = 0.125 m            initial core radius a_0 (radius of PEAK
#                                      azimuthal velocity — C&W convention)
#   $a_0/b_0$     = 0.125              core-to-separation ratio
#   $\sigma_0 = a_0/1.12091$          = 0.1115 m   diffused Gaussian width
#
# C&W's a_0 is the radius of maximum azimuthal velocity, which for a Lamb-Oseen
# vortex is r_max = 1.12091*sigma (root of e^x = 1 + 2x). LambOseenVPM diffuses
# the Gaussian width sigma = a_0/1.12091 -- the --core-radius / --separation
# values below are the C&W peak-velocity radii and are NOT rescaled here; the
# conversion lives in vortex_setup.py (and assets/_common.py for the plots).
#
# Derived reference quantities:
#   $Re = \Gamma/\nu$                    = 530        Reynolds number
#   $U_{c,0} = \Gamma/(2\pi r_{c,0})$   = 1.273 m/s  reference velocity
#   $\omega_{c,0} = \Gamma/(\pi r_{c,0}^2)$ = 20.37 1/s  reference vorticity
#   $G_{c,0} = U_{c,0}/r_{c,0}$        = 10.19 1/s  reference gradient
#   $t_0 = \sigma_0^2/(4\nu)$          = 1.65 s     initial vortex age
#   $h$ (particle spacing)              = 0.04125 m  (0.33 * r_{c,0})
#
# Co-rotating merger runs override the legacy tutorial core radius with
# $a_0/b_0 = 0.17$, matching the C&W merger benchmark.
#
# Buckingham-Pi groups used in figures:
#   $\tau = \nu t / b_0^2$                              time
#   $r^* = r / r_{c,0}$                                 radius
#   $u^* = u_\theta / U_{c,0}$                          azimuthal velocity
#   $\omega^* = \omega_z / \omega_{c,0}$                vorticity
#   $G^* = (\partial u_y/\partial x)\,r_{c,0}/U_{c,0}$  velocity gradient
#   $x_c^* = x_c / b_0$                                 dipole trajectory
#   $r_c^* = r_c / r_{c,0}$                             core radius (dipole)
#   $\sigma^2 / b_0^2$                                  core area (merging)
#   $b^* = b / b_0$                                     separation (merging)
#   $P^* = (dE/dt) / (\nu\Gamma^2/b_0^2)$               energy dissipation
# ─────────────────────────────────────────────────────────────────────────────

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

COMMON_FLAGS=(--re 530 --dt 0.04 --solution-dir ./solution --clean --processing-unit CUDA)
CS_RWM_FLAGS=(--spacing-factor 0.4 --length 20 --total-time 20.0)
DVH_FLAGS=(--grid-spacing-factor 0.40899787483419064 --dvh-max-nodes 120000 --length 16 --total-time 20.0)
GBD_FLAGS=(--grid-spacing-factor 0.4 --length 16 --total-time 20.0)
MERGING_GEOMETRY_FLAGS=(--core-radius 0.17 --separation 1.0)
MERGING_DVH_FLAGS=(--grid-spacing-factor 0.3007337314957284 --dvh-max-nodes 120000 --length 16 --total-time 20.0)

# ============================================================
# TEST CASE 1: Single Lamb-Oseen vortex diffusion
# ============================================================
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes cs \
    "${COMMON_FLAGS[@]}" "${CS_RWM_FLAGS[@]}"

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes rwm \
    "${COMMON_FLAGS[@]}" "${CS_RWM_FLAGS[@]}"

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes dvh \
    "${COMMON_FLAGS[@]}" "${DVH_FLAGS[@]}"

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 0.0 --schemes gbd \
    "${COMMON_FLAGS[@]}" "${GBD_FLAGS[@]}"

sleep 15

echo 'DONE WITH SINGLE VORTEX'

# ============================================================
# TEST CASE 2: Vortex dipole (counter-rotating pair)
# ============================================================
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes cs \
    "${COMMON_FLAGS[@]}" --spacing-factor 0.40 --total-time 20.0

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes rwm \
    "${COMMON_FLAGS[@]}" --spacing-factor 0.40 --total-time 20.0

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes dvh \
    "${COMMON_FLAGS[@]}" "${DVH_FLAGS[@]}"

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 -1.0 --schemes gbd \
    "${COMMON_FLAGS[@]}" "${GBD_FLAGS[@]}"


echo 'DONE WITH DIPOLE'


sleep 15

# ============================================================
# TEST CASE 3: Co-rotating vortex merger vortex merger
# ============================================================
"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes cs \
    "${COMMON_FLAGS[@]}" "${MERGING_GEOMETRY_FLAGS[@]}" --spacing-factor 0.40 --length 20 --total-time 20.0

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes rwm \
    "${COMMON_FLAGS[@]}" "${MERGING_GEOMETRY_FLAGS[@]}" --spacing-factor 0.40 --length 20 --total-time 20.0

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes dvh \
    "${COMMON_FLAGS[@]}" "${MERGING_GEOMETRY_FLAGS[@]}" "${MERGING_DVH_FLAGS[@]}"

"$PYTHON" vortex_setup.py \
    --gamma1 1.0 --gamma2 1.0 --schemes gbd \
    "${COMMON_FLAGS[@]}" "${MERGING_GEOMETRY_FLAGS[@]}" "${GBD_FLAGS[@]}"
