#!/bin/sh
# Remove generated coupled output while preserving reference_flow/.
cd "$(dirname "$0")" || exit 1

rm -rf solution samples constant figures .matplotlib .cache .taichi_cache
rm -rf __pycache__ assets/__pycache__ reference_flow/__pycache__
rm -f ./*.log .openonda_run.lock

echo "Cleaned root solution/, samples/, constant/, figures/, and caches; preserved reference_flow/."
