#!/bin/sh
set -eu

cd "$(dirname "$0")"

./allclean.sh
python -u cubeFlow_setup.py
python assets/check_run.py
