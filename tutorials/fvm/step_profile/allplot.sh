#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m assets.plot_profile
python -m assets.plot_comparison
