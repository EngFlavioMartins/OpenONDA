#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

# Keep results; setup.py resumes the current native checkpoint.
python setup.py "$@"
