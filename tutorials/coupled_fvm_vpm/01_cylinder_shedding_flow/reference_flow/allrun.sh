#!/bin/bash
set -e

cd "$(dirname "$0")"

# The solver of the campaign, as the coupled runs (allrun.sh one level up): the frozen export of
# commit 3f771630
export PYTHONPATH=${STUDY_SOLVER:-$HOME/openonda_runs/study_2026-11/solver_3f771630}

# Remove previous output and simulate every case from t = 0.
./allclean.sh

# Grid independence at Re = 150: four cell sizes, the step refined with them
# (dt = 0.008 s h / 0.04 m). h = 0.04 m is the coupled baseline.
python setup.py --cell-size 0.08 --name h0.08_dt0.016
python setup.py --cell-size 0.06 --name h0.06_dt0.012
python setup.py --cell-size 0.04 --name h0.04_dt0.008
python setup.py --cell-size 0.03 --name h0.03_dt0.006
