#!/bin/bash
set -e

# Usage: ./allplot.sh [png|pdf|both] (default: both)
cd "$(dirname "$0")"

python -m assets.plot_ring_speed --format "${1:-both}" --output figures --auxiliary-output figures/auxiliary
python -m assets.plot_core_trajectories dns les les_stabilized --merge-bridge .9 --format "${1:-both}" --output figures --auxiliary-output figures/auxiliary
python -m assets.plot_core_sections --runs dns les les_stabilized --times 1.05 3.45 5.25 6.75 7.5 9 12 --format "${1:-both}" --output figures --auxiliary-output figures/auxiliary
python -m assets.plot_diagnostic_histories dns les les_stabilized --format "${1:-both}" --output figures
python -m assets.plot_group_history dns les les_stabilized --format "${1:-both}" --output figures
python -m assets.plot_diagnostic_histories high_re_dns high_re_les --figure-name high_re_histories --format "${1:-both}" --output figures
