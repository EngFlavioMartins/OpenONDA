#!/bin/bash -e
cd -- "$(dirname -- "$0")"

./allclean.sh

python setup.py --mode moving --angle -10
python setup.py --mode moving --angle -5
python setup.py --mode moving --angle -2
python setup.py --mode moving --angle 0
python setup.py --mode moving --angle 2
python setup.py --mode moving --angle 5
python setup.py --mode moving --angle 8
python setup.py --mode moving --angle 10
python setup.py --mode moving --angle 12
python setup.py --mode moving --angle 15
python setup.py --mode static --angle -10
python setup.py --mode static --angle -5
python setup.py --mode static --angle -2
python setup.py --mode static --angle 0
python setup.py --mode static --angle 2
python setup.py --mode static --angle 5
python setup.py --mode static --angle 8
python setup.py --mode static --angle 10
python setup.py --mode static --angle 12
python setup.py --mode static --angle 15
