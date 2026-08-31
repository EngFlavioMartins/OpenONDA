#!/bin/sh
# Remove generated conventional-FVM output while retaining convergence evidence.
cd "$(dirname "$0")" || exit 1

if [ -d solution ]; then
    find solution -mindepth 1 -maxdepth 1 ! -name verification -exec rm -rf -- {} +
fi
rm -rf samples constant .matplotlib __pycache__
rm -f ./*.log .openonda_run.lock

echo "Cleaned generated reference output; preserved solution/verification/."
