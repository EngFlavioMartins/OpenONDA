#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove previous output and simulate the fixed cylinders from t = 0.
./allclean.sh
python setup.py "$@"
