#!/bin/sh
# Remove all generated output from the referenceFlow case.
cd "$(dirname "$0")" || exit 1

rm -rf solution constant __pycache__ assets/__pycache__
rm -f ./*.log
rm -f assets/mesh.msh

echo "Cleaned: solution/ constant/ caches and logs removed."
