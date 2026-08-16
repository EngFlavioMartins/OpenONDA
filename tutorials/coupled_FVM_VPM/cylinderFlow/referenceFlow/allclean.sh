#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
rm -rf solution constant samples __pycache__
find . -name '*.pyc' -delete
echo "Cleaned reference solution, samples, and caches."
