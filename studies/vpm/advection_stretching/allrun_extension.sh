#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="studies.vpm.advection_stretching.assets"

python -m "${MODULE}.run_formulation_comparison"
for configuration in \
    exact_pair_rk3_isolated \
    tree_gradient_rk3_isolated \
    production_numerics_unforced
do
    python -m "${MODULE}.run_full_checkpoint" \
        --checkpoint leapfrog --configuration "${configuration}" --steps 2
    python -m "${MODULE}.run_scale_timing" --configuration "${configuration}"
done

# The full rotor pairwise replay is intentionally absent: run_scale_timing.py
# records the measured feasibility-gate decision before these feasible arms.
for configuration in tree_gradient_rk3_isolated production_numerics_unforced
do
    python -m "${MODULE}.run_full_checkpoint" \
        --checkpoint rotor --configuration "${configuration}" --steps 2
done

python -m "${MODULE}.summarize_extension"
