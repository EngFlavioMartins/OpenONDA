#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue cases B1 and B2 from their latest accepted coupled backups.
# B1
python setup.py --reynolds-number 6400 --reduced-velocity 7.68 "$@"
# B2
python setup.py --reynolds-number 8300 --reduced-velocity 9.96 "$@"
