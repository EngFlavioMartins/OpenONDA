#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove previous output and simulate the paper's cases B1 and B2 from t = 0.
./allclean.sh
# B1
python setup.py --reynolds-number 6400 --reduced-velocity 7.68 "$@"
# B2
python setup.py --reynolds-number 8300 --reduced-velocity 9.96 "$@"
