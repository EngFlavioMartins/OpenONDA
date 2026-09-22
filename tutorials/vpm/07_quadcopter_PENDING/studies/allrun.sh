#!/bin/bash -e
cd -- "$(dirname -- "$0")"

./allclean.sh

python setup.py --case coarse
python setup.py --case time_refined
python setup.py --case mesh_refined
python setup.py --case relaxed
python setup.py --case relaxed_time_refined
python setup.py --case continue_8
python setup.py --case continue_12
