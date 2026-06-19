#!/usr/bin/env bash
# =============================================================================
# allrun.sh — Flat Plate VLM-VPM Certification Suite
# =============================================================================
#
# Part A — AoA polar sweep (10 points, −10° … 15°)
# -------------------------------------------------
#   Moving wing, body frame, sin²-ramp start, Kutta-Joukowski forces.
#   Produces the CL/CD/CM polar and the spanwise distribution figure.
#
#   α [°]: −10  −5  −2   0   2   5   8  10  12  15
#
# Part B — Wind-frame static polar sweep (10 points, same AoA set as Part A)
# ---------------------------------------------------------------------------
#   Static plate, wind frame — validates the Kutta-Joukowski formulation
#   against the moving-wing result across the full polar.
#
# Common parameters
# -----------------
#   Geometry : chord = 1 m,  span = 10 m  (AR = 10),  8×16 panels
#   Physics  : U = 10 m/s,  ρ = 1 kg/m³,  ν = 0.01 m²/s
#   Time     : τ_ramp = 0.6 c,  τ_max = 30 c,  dt = 0.01 s
#
# Usage
# -----
#   ./allrun.sh                    # run everything
#   ./allrun.sh 2>&1 | tee run.log
#
# Results land in  solution/exp_*/samples/exp_*.csv
# Figures saved to figures/
#
# Author:  Flavio A. C. Martins, OpenONDA Team
# Date: June 2026
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./allclean.sh

PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null || command -v python3 || command -v python)"
RUN="$PYTHON setup_plate.py"

# Common flags shared by every moving-wing run
COMMON="--kinematics ramp --frame body
        --chord 1 --span 10 --panels-chord 8 --panels-span 16
        --tau-ramp 0.6 --tau-max 30 --dt 0.01"

# Common flags shared by every static wind-frame run (no ramp)
STATIC="--kinematics static --frame wind
        --chord 1 --span 10 --panels-chord 8 --panels-span 16
        --tau-max 30 --dt 0.01"

echo "========================================================"
echo "  OpenONDA Flat Plate — Certification Suite"
echo "========================================================"
echo ""

# =============================================================================
# PART A — AoA polar sweep (moving wing, body frame)
# =============================================================================
echo "--- Part A: AoA polar sweep ---"
echo ""

echo "  [A01] α = -10°"
$RUN --name exp_moving_aoan10 $COMMON --aoa=-10
echo ""

echo "  [A02] α = -5°"
$RUN --name exp_moving_aoan05 $COMMON --aoa=-5
echo ""

echo "  [A03] α = -2°"
$RUN --name exp_moving_aoan02 $COMMON --aoa=-2
echo ""

echo "  [A04] α = 0°"
$RUN --name exp_moving_aoa00 $COMMON --aoa=0
echo ""

echo "  [A05] α = 2°"
$RUN --name exp_moving_aoa02 $COMMON --aoa=2
echo ""

echo "  [A06] α = 5°"
$RUN --name exp_moving_aoa05 $COMMON --aoa=5
echo ""

echo "  [A07] α = 8°"
$RUN --name exp_moving_aoa08 $COMMON --aoa=8
echo ""

echo "  [A08] α = 10°"
$RUN --name exp_moving_aoa10 $COMMON --aoa=10
echo ""

echo "  [A09] α = 12°"
$RUN --name exp_moving_aoa12 $COMMON --aoa=12
echo ""

echo "  [A10] α = 15°"
$RUN --name exp_moving_aoa15 $COMMON --aoa=15
echo ""

# =============================================================================
# PART B — Wind-frame static polar sweep (matching Part A AoA set)
# =============================================================================
echo "--- Part B: Wind-frame static polar sweep ---"
echo ""

echo "  [B01] α = -10°"
$RUN --name exp_static_aoan10 $STATIC --aoa=-10
echo ""

echo "  [B02] α = -5°"
$RUN --name exp_static_aoan05 $STATIC --aoa=-5
echo ""

echo "  [B03] α = -2°"
$RUN --name exp_static_aoan02 $STATIC --aoa=-2
echo ""

echo "  [B04] α = 0°"
$RUN --name exp_static_aoa00 $STATIC --aoa=0
echo ""

echo "  [B05] α = 2°"
$RUN --name exp_static_aoa02 $STATIC --aoa=2
echo ""

echo "  [B06] α = 5°"
$RUN --name exp_static_aoa05 $STATIC --aoa=5
echo ""

echo "  [B07] α = 8°"
$RUN --name exp_static_aoa08 $STATIC --aoa=8
echo ""

echo "  [B08] α = 10°"
$RUN --name exp_static_aoa10 $STATIC --aoa=10
echo ""

echo "  [B09] α = 12°"
$RUN --name exp_static_aoa12 $STATIC --aoa=12
echo ""

echo "  [B10] α = 15°"
$RUN --name exp_static_aoa15 $STATIC --aoa=15
echo ""

# =============================================================================
# Post-processing
# =============================================================================
echo "========================================================"
echo "  All experiments complete.  Generating figures..."
echo "========================================================"
echo ""

./allplot.sh

echo ""
echo "========================================================"
echo "  Done.  Figures saved to figures/"
echo "========================================================"
