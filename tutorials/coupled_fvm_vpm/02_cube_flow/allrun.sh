#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

# Preserve outputs; setup.py resumes only a compatible native backup.
python setup.py "$@"
