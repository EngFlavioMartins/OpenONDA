#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare the vibration of the elastically mounted cylinder with the paper and the FVM reference.
python -m assets.plot_response
python -m assets.plot_histories
python -m assets.render_vorticity
python -m assets.plot_coupling_cost
