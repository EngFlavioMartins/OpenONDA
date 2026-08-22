#!/usr/bin/env bash
# Run the coupled cube-flow case.
set -euo pipefail

cd "$(dirname "$0")"

if [ -d samples ]; then
    sample_file="$(find samples -type f -print -quit)"
    if [ -n "$sample_file" ]; then
        archive="run_backups/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$archive"
        mv samples "$archive/samples"
        echo "Archived previous samples in $archive/samples"
    fi
fi

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
mkdir -p solution
python -u cubeFlow_setup.py 2>&1 | tee solution/cube_flow.log
