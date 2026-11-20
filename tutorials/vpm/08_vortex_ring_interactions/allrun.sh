#!/bin/bash
set -e

cd "$(dirname "$0")"

./allclean.sh

python setup.py collision_les_cusp
python setup.py collision_les
python setup.py collision_dns
python setup.py leapfrog_les_cusp
python setup.py leapfrog_les
python setup.py leapfrog_dns
