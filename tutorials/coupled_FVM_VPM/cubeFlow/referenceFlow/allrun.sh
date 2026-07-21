#!/bin/sh
set -eu

cd "$(dirname "$0")" || exit 1

./allclean.sh
python ./assets/create_mesh.py > mesh.log
python referenceFlow_setup.py
