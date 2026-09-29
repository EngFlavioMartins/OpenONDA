#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python assets/run_pipeline.py --run-dir study_results/cylinder/default --resume
