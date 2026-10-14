#!/bin/bash
set -e
cd "$(dirname "$0")/.."  # the case directory
python -m assets.plot_sensitivity "$@"
