#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python assets/run_formulation_comparison.py
for configuration in \
    exact_pair_rk3_isolated \
    tree_gradient_rk3_isolated \
    production_numerics_unforced
do
    python assets/run_full_checkpoint.py \
        --checkpoint leapfrog --configuration "${configuration}" --steps 2
    python assets/run_scale_timing.py --configuration "${configuration}"
done

# The full rotor pairwise replay is intentionally absent: run_scale_timing.py
# records the measured feasibility-gate decision before these feasible arms.
for configuration in tree_gradient_rk3_isolated production_numerics_unforced
do
    python assets/run_full_checkpoint.py \
        --checkpoint rotor --configuration "${configuration}" --steps 2
done

python assets/summarize_extension.py
