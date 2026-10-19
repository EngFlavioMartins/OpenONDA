#!/bin/bash
set -e
cd "$(dirname "$0")"
python -m assets.plot_forces "$@"
python -m assets.plot_velocity_profiles "$@"
python -m assets.plot_fields "$@"
python -m assets.plot_coupling_cost "$@"
