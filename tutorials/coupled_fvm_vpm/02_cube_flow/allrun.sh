#!/bin/bash
set -e

cd "$(dirname "$0")"

# Resume the latest native checkpoint.
python -m openonda.tutorial_runner . setup "$@"
