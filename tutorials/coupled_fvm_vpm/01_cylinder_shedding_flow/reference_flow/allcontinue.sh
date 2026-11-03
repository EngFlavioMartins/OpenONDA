#!/bin/bash
set -e

cd "$(dirname "$0")"

# The solver of the campaign, as the coupled runs (allrun.sh one level up): the frozen export of
# commit 3f771630
export PYTHONPATH=${STUDY_SOLVER:-$HOME/openonda_runs/study_2026-11/solver_3f771630}

# Continue every case of allrun.sh from its latest accepted FVM backup.
python setup.py --cell-size 0.08 --name h0.08_dt0.016
python setup.py --cell-size 0.06 --name h0.06_dt0.012
python setup.py --cell-size 0.04 --name h0.04_dt0.008
python setup.py --cell-size 0.03 --name h0.03_dt0.006
