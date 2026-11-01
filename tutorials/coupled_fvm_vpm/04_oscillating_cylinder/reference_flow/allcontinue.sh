#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue from the latest accepted FVM backup.
python setup.py "$@"
