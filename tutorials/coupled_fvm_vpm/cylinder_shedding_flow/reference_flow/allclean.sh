#!/bin/sh
# Remove all generated output from the reference_flow case.
cd "$(dirname "$0")" || exit 1

rm -rf solution constant samples figures .matplotlib
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log

echo "Cleaned: reference_flow solution/ constant/ samples/ figures, caches, and logs."