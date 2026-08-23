#!/bin/sh
set -eu

cd "$(dirname "$0")"
rm -rf solution constant samples figures .matplotlib __pycache__ assets/__pycache__
rm -f ./*.log
echo "Cleaned generated NACA 4412 FVM-VPM output."
