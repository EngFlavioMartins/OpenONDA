#!/bin/bash -e

python setup.py --variant dns_direct
python setup.py --variant dns_transposed
python setup.py --variant dns_mixed
python setup.py --variant les_transposed
