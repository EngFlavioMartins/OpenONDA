#!/bin/sh
set -eu

cd "$(dirname "$0")" || exit 1

./allclean.sh
python run_setup.py
