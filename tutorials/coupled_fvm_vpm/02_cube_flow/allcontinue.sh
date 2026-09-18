#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py --restart-from solution/backups
