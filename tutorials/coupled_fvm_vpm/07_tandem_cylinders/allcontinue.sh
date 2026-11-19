#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue from the latest accepted coupled backup.
python setup.py "$@"
