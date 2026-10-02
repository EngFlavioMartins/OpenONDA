#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

python setup.py -h 0.04 "$@"
