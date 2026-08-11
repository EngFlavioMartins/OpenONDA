#!/usr/bin/env bash
# Run every vortex-ring physics case and make the comparison figures.

set -euo pipefail

cd "$(dirname "$0")"
./allclean.sh

SAMPLE_PERIOD=0.1
BACKUP_PERIOD=0.5

run() {
    python ring_setup.py "$1" "$SAMPLE_PERIOD" "$BACKUP_PERIOD"
}

run DNS_direct
run DNS_transposed
run DNS_mixed
run LES_transposed

./allplot.sh
