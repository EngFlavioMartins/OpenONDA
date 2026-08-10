#!/bin/sh
set -eu

cd "$(dirname "$0")"
python assets/plot_forces.py
