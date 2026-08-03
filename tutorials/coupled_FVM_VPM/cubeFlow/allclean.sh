#!/bin/sh
# Remove all generated output from the native FVM+VPM cube case.
cd "$(dirname "$0")" || exit 1

rm -rf solution constant figures .matplotlib
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log

echo "Cleaned: solution/ constant/ figures, caches, and logs removed."
