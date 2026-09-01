#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python assets/run_manufactured.py
python assets/run_discrete_clouds.py
python assets/run_checkpoint_replay.py
python assets/run_production_envelope.py
python assets/run_performance.py
python assets/plot_results.py
