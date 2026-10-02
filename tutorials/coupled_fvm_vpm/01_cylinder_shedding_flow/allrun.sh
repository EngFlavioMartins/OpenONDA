#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

# Keep existing results; setup.py resumes a compatible native backup.
python setup.py "$@"
