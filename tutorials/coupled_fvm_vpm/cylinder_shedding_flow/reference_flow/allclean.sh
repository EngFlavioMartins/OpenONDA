#!/bin/sh
# Remove the generated conventional-FVM reference output.
cd "$(dirname "$0")" || exit 1

rm -rf solution samples constant .matplotlib __pycache__
rm -f ./*.log .openonda_run.lock

echo "Cleaned reference_flow/solution, reference_flow/samples, and caches."
