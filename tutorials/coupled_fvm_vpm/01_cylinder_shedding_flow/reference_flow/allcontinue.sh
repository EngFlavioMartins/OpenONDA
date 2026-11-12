#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue every case of allrun.sh from its latest accepted FVM backup; a case without one starts
# from t = 0.
python setup.py --cell-size 0.08 --name h0.08_dt0.016
python setup.py --cell-size 0.06 --name h0.06_dt0.012
python setup.py --cell-size 0.04 --name h0.04_dt0.008
python setup.py --cell-size 0.03 --name h0.03_dt0.006
python setup.py --cell-size 0.04 --fine-box-margin 3.2 --end-time 55 --name h0.04_dt0.008_fb3.2
python setup.py --cell-size 0.04 --fine-box-margin 4.0 --end-time 55 --name h0.04_dt0.008_fb4.0
