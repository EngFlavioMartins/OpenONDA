#!/bin/bash
set -e

cd "$(dirname "$0")"

python setup.py collision_les
python setup.py collision_dns
python setup.py leapfrog_les
python setup.py leapfrog_dns
