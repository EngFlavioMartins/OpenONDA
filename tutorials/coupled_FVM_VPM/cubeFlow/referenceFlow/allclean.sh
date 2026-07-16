#!/bin/sh
# Remove all generated output from the referenceFlow case.
cd "$(dirname "$0")" || exit 1

rm -rf solution __pycache__ assets/__pycache__
rm -f ./*.log

echo "Cleaned: solution/ caches and logs removed."
