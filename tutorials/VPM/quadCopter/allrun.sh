#!/usr/bin/env bash
# Run the quadcopter case and make its figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.0375
BACKUP_PERIOD=0.0125  # 24 animation frames per rotor revolution.

python quad_setup.py "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
./allplot.sh
