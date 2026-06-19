#!/usr/bin/env bash
# Vortex-ring interactions — generate the stabilization comparison figures.
#
# Usage:
#   ./allplot.sh [--solution-dir PATH] [--dpi N]
#
# Defaults: --solution-dir ./solution  --dpi 300
#
# Case directories under solution-dir/ are auto-discovered (any name with
# vpm_*_NNNNNN.h5 backups) — see assets/compare_stabilization.py.

set -euo pipefail

PYTHON="$(conda run -n OpenONDA which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOLUTION_DIR="./solution"
DPI=300
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

FIGURES_DIR="$SCRIPT_DIR/figures"
mkdir -p "$FIGURES_DIR"

echo "[allplot] solution-dir : $SOLUTION_DIR"
echo "[allplot] figures-dir  : $FIGURES_DIR"
echo "[allplot] dpi          : $DPI"
echo ""

$PYTHON assets/compare_stabilization.py \
    --solution-dir "$SOLUTION_DIR" \
    --figures-dir  "$FIGURES_DIR" \
    --dpi          "$DPI"

echo ""
echo "[allplot] Done. Figures saved to $FIGURES_DIR"
