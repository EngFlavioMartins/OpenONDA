#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py --name dense --dx 0.040 --end-time 100.0 --restart-from solution/dense/backup
