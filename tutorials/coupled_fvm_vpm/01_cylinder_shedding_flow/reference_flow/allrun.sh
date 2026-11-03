#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove previous output and simulate every case from t = 0 (--continue: resume them instead).
[[ ${1:-} == --continue ]] || ./allclean.sh

# Flow configurations at the baseline cells
python setup.py --reynolds-number 100 --cell-size 0.04 --name re100_h0.04
python setup.py --reynolds-number 150 --cell-size 0.04 --name re150_h0.04

# Grid sensitivity at the highest Reynolds number, cells refined by sqrt(2)
python setup.py --reynolds-number 250 --cell-size 0.08 --name re250_h0.08
python setup.py --reynolds-number 250 --cell-size 0.056569 --name re250_h0.056569
python setup.py --reynolds-number 250 --cell-size 0.04 --name re250_h0.04
python setup.py --reynolds-number 250 --cell-size 0.028284 --name re250_h0.028284
