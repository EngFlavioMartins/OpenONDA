#!/usr/bin/env bash
# Run the rotor-wake case, plot it, and check the sampled results.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.12
BACKUP_PERIOD=0.03  # About 26 animation frames per rotor revolution.

python rotor_setup.py "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
./allplot.sh
python assets/validate_results.py
