#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . assets.plot_forces --format "${1:-both}"
