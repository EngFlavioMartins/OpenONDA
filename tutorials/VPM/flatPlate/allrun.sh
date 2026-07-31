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
#   Geometry : chord = 1 m,  span = 10 m  (AR = 10),  8×14 panels
#   Physics  : U = 10 m/s,  ρ = 1 kg/m³,  ν = 0.01 m²/s
#   Time     : τ_ramp = 0.6 c,  τ_max = 24 c,  dt = 0.0125 s
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
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./allclean.sh

FAILED=0

run_case() {
    local name=""
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--name" && $((i+1)) -le $# ]]; then
            j=$((i+1))
            name="${!j}"
            break
        fi
    done
    local log_dir="solution/${name}"
    mkdir -p "$log_dir"
    echo "  Log: ${log_dir}/${name}.log"
    local rc=0
    python setup_plate.py "$@" > "${log_dir}/${name}.log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  *** ${name} exited with code ${rc}" >&2
        FAILED=1
    fi
}

# Common flags shared by every moving-wing run
COMMON="--kinematics ramp --frame body
        --chord 1 --span 10 --panels-chord 8 --panels-span 14
        --tau-ramp 0.6 --tau-max 24 --dt 0.0125
        --processing-unit CUDA"

# Common flags shared by every static wind-frame run (no ramp)
STATIC="--kinematics static --frame wind
        --chord 1 --span 10 --panels-chord 8 --panels-span 14
        --tau-max 24 --dt 0.0125
        --processing-unit CUDA"

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
run_case --name exp_moving_aoan10 $COMMON --aoa=-10
echo ""

echo "  [A02] α = -5°"
run_case --name exp_moving_aoan05 $COMMON --aoa=-5
echo ""

echo "  [A03] α = -2°"
run_case --name exp_moving_aoan02 $COMMON --aoa=-2
echo ""

echo "  [A04] α = 0°"
run_case --name exp_moving_aoa00 $COMMON --aoa=0
echo ""

echo "  [A05] α = 2°"
run_case --name exp_moving_aoa02 $COMMON --aoa=2
echo ""

echo "  [A06] α = 5°"
run_case --name exp_moving_aoa05 $COMMON --aoa=5
echo ""

echo "  [A07] α = 8°"
run_case --name exp_moving_aoa08 $COMMON --aoa=8
echo ""

echo "  [A08] α = 10°"
run_case --name exp_moving_aoa10 $COMMON --aoa=10
echo ""

echo "  [A09] α = 12°"
run_case --name exp_moving_aoa12 $COMMON --aoa=12
echo ""

echo "  [A10] α = 15°"
run_case --name exp_moving_aoa15 $COMMON --aoa=15
echo ""

# =============================================================================
# PART B — Wind-frame static polar sweep (matching Part A AoA set)
# =============================================================================
echo "--- Part B: Wind-frame static polar sweep ---"
echo ""

echo "  [B01] α = -10°"
run_case --name exp_static_aoan10 $STATIC --aoa=-10
echo ""

echo "  [B02] α = -5°"
run_case --name exp_static_aoan05 $STATIC --aoa=-5
echo ""

echo "  [B03] α = -2°"
run_case --name exp_static_aoan02 $STATIC --aoa=-2
echo ""

echo "  [B04] α = 0°"
run_case --name exp_static_aoa00 $STATIC --aoa=0
echo ""

echo "  [B05] α = 2°"
run_case --name exp_static_aoa02 $STATIC --aoa=2
echo ""

echo "  [B06] α = 5°"
run_case --name exp_static_aoa05 $STATIC --aoa=5
echo ""

echo "  [B07] α = 8°"
run_case --name exp_static_aoa08 $STATIC --aoa=8 --sample-crossflow
echo ""

echo "  [B08] α = 10°"
run_case --name exp_static_aoa10 $STATIC --aoa=10
echo ""

echo "  [B09] α = 12°"
run_case --name exp_static_aoa12 $STATIC --aoa=12
echo ""

echo "  [B10] α = 15°"
run_case --name exp_static_aoa15 $STATIC --aoa=15
echo ""

# =============================================================================
# Post-processing
# =============================================================================
echo ""
if [[ "$FAILED" -ne 0 ]]; then
    echo "One or more flat-plate simulations failed." >&2
    exit 1
fi
echo "========================================================"
echo "  All experiments complete.  Generating figures..."
echo "========================================================"
echo ""

./allplot.sh

echo ""
echo "========================================================"
echo "  Done.  Figures saved to figures/"
echo "========================================================"
