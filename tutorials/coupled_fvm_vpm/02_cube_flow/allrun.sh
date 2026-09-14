#!/bin/bash -e
cd -- "$(dirname -- "$0")"
./allclean.sh --keep-mesh
python setup.py
