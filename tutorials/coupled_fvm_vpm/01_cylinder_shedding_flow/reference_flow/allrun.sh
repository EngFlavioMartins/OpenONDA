#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py --name "grid_h008" -h 0.08
python setup.py --name "grid_h00565685" -h 0.0565685424949238
python setup.py --name "grid_h004" -h 0.04
python setup.py --name "grid_h00282843" -h 0.0282842712474619
