#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

# One selected mesh. Preserve completed results and resume native backups.
python setup.py -h 0.04 "$@"
