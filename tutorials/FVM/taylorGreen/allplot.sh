#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOLUTION_DIR="./solution"
DPI=300
FORMAT="png"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solution-dir) SOLUTION_DIR="$2"; shift 2 ;;
        --dpi)          DPI="$2";          shift 2 ;;
        --format)       FORMAT="$2";       shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p figures
export MPLCONFIGDIR="$SOLUTION_DIR/.matplotlib"
export XDG_CACHE_HOME="$SOLUTION_DIR/.cache"
python assets/plot_decay.py \
    --history "$SOLUTION_DIR/history.csv" \
    --output "figures/taylor_green_decay.$FORMAT" \
    --dpi "$DPI"
echo "[allplot] Figure saved to $SCRIPT_DIR/figures"
