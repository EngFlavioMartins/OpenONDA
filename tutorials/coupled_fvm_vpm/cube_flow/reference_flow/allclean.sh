#!/bin/sh
# Remove all generated output from the reference_flow case.
cd "$(dirname "$0")" || exit 1

rm -rf solution constant __pycache__ assets/__pycache__
rm -f ./*.log

echo "Cleaned: solution/ constant/ caches and logs removed."
