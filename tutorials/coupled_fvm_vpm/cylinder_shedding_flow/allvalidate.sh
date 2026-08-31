#!/usr/bin/env bash
# Validate existing cylinder output. This script never launches or cleans a case.
set -euo pipefail

cd "$(dirname "$0")"
mode="${1:-reference}"
case "$mode" in
    reference|all) ;;
    *) echo "Usage: ./allvalidate.sh [reference|all]" >&2; exit 2 ;;
esac

verification="reference_flow/solution/verification"

echo
echo "===== REFERENCE DIAGNOSTICS ====="
echo
python assets/analyse_reference.py --require-ready reference_flow
python assets/audit_reference_samples.py reference_flow
python assets/check_grid_independence.py \
    --g0 "$verification/g0_forces_history.csv" \
    --g1 reference_flow/samples/forces_history.csv \
    --g2 "$verification/g2_forces_history.csv" \
    --half-dt "$verification/g1_half_dt_forces_history.csv" \
    --large-domain "$verification/g1_large_domain_forces_history.csv"

if [ "$mode" = "all" ]; then
    echo
    echo "===== COUPLED DIAGNOSTICS ====="
    echo
    python assets/check_run.py
    python assets/analyse_coupled_benchmark.py
    python assets/analyse_von_karman.py
fi

./allplot.sh png
./allplot.sh pdf

echo
echo "===== VALIDATION PASSED ($mode) ====="
