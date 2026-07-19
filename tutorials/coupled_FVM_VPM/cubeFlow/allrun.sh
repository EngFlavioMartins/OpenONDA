#!/bin/sh
cd "$(dirname "$0")" || exit 1
./allclean.sh
python -u cube_setup.py
./allplot.sh
