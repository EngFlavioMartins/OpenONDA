#!/usr/bin/env bash
# Run the coupled cube-flow case.
set -euo pipefail

cd "$(dirname "$0")"

previous_output=""
for directory in samples solution figures; do
    if [ -d "$directory" ] && [ -n "$(find "$directory" -type f -print -quit)" ]; then
        previous_output=1
        break
    fi
done

if [ -n "$previous_output" ]; then
    archive="run_backups/$(date -u +%Y%m%dT%H%M%SZ)"
    suffix=1
    while [ -e "$archive" ]; do
        archive="${archive%_*}_$suffix"
        suffix=$((suffix + 1))
    done
    mkdir -p "$archive"
    for directory in samples solution figures; do
        if [ -d "$directory" ]; then
            mv "$directory" "$archive/$directory"
        fi
    done
    cp cubeFlow_setup.py "$archive/cubeFlow_setup.py"
    echo "Archived previous run in $archive"
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
