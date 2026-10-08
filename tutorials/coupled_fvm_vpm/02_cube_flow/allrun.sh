#!/bin/bash
set -e

# Resume the latest native checkpoint.
cd "$(dirname "$0")"

./allclean.sh

python setup.py "$@"
