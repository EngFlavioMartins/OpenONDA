#!/bin/sh
# Remove all generated output from the cylinder FVM-VPM case (both the coupled
# case and its fully meshed reference), including archived runs.
cd "$(dirname "$0")" || exit 1

rm -rf solution constant samples figures runs .matplotlib
rm -rf __pycache__ assets/__pycache__ reference_flow/solution reference_flow/constant reference_flow/__pycache__
rm -f ./*.log reference_flow/*.log

echo "Cleaned: solution/ constant/ samples/ figures/ runs/ reference_flow output, caches, and logs."