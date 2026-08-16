#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
rm -rf solution constant samples figures __pycache__ assets/__pycache__ .matplotlib
find . -name '*.pyc' -delete
echo "Cleaned hybrid solution, samples, figures, and caches; referenceFlow was preserved."
