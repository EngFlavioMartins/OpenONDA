#!/bin/sh
cd "$(dirname "$0")" || exit 1
rm -rf solution samples figures .matplotlib __pycache__ assets/__pycache__
rm -f ./*.log
echo "Cleaned: solution/ samples/ figures/, caches and logs removed."
