#!/bin/bash -e
cd -- "$(dirname -- "$0")"

./allclean.sh

python setup.py --variant dns_direct
python setup.py --variant dns_transposed
python setup.py --variant dns_mixed
python setup.py --variant les_transposed
