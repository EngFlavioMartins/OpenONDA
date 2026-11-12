#!/bin/bash
set -e

cd "$(dirname "$0")"

# State of every study run, then the statistics, errors and figures of the calibration study,
# written to figures/ (PDF and PNG) with figures/study_summary.csv, and the phase-averaged field
# comparison of the baseline. Figures whose runs are not finished yet are skipped with a message.
python -m assets.study_status
python -m assets.plot_study "$@"
python -m assets.plot_fields
