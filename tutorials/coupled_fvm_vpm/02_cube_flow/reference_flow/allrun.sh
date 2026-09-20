#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py --name "grid_h010125" -h 0.10125
python setup.py --name "grid_h00675" -h 0.0675
python setup.py --name "grid_h0045" -h 0.045
python setup.py --name "grid_h003" -h 0.03
