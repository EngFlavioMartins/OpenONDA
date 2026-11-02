#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue each reduced velocity from its latest accepted backup.
python setup.py --reduced-velocity 4 "$@"
python setup.py --reduced-velocity 6 "$@"
python setup.py --reduced-velocity 8 "$@"
