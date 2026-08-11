#!/usr/bin/env bash
# Run the delta-wing wake-crossing case and make its figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.08
BACKUP_PERIOD=0.04  # 25 animation frames per heave cycle.

python delta_wing_setup.py "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
./allplot.sh
