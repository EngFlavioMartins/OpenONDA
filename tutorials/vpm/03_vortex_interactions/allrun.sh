#!/bin/bash
set -e

cd "$(dirname "$0")"

./allclean.sh

python setup.py single_ring
python setup.py dns
python setup.py les
python setup.py les_stabilized
python setup.py high_re_dns
python setup.py high_re_les
