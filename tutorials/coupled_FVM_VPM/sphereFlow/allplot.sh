#!/usr/bin/env bash
# Compare the hybrid against the fully meshed reference.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR" figures
format="${1:-png}"
case "$format" in png|pdf) ;; *) echo "Usage: $0 [png|pdf]" >&2; exit 2 ;; esac
python assets/compare_unsteady.py --format "$format"
