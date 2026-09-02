#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="studies.vpm.advection_stretching.assets"
python -m "${MODULE}.run_manufactured"
python -m "${MODULE}.run_discrete_clouds"
python -m "${MODULE}.run_checkpoint_replay"
python -m "${MODULE}.run_production_envelope"
python -m "${MODULE}.run_performance"
python -m "${MODULE}.plot_results"
