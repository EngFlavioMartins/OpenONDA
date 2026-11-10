#!/bin/bash
set -e

cd "$(dirname "$0")"

# Statistics, coupling errors and every figure of the study, written to figures/
python -m assets.plot_study
