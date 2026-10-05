#!/bin/bash
set -e

cd "$(dirname "$0")"

# Preserve each grid's existing results and resume its compatible native backup.
python -m openonda.tutorial_runner . setup --name "grid_h010125" -h 0.10125
python -m openonda.tutorial_runner . setup --name "grid_h00675" -h 0.0675
python -m openonda.tutorial_runner . setup --name "grid_h0045" -h 0.045
