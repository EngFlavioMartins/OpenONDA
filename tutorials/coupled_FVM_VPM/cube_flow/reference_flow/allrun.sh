#!/bin/sh
set -eu

cd "$(dirname "$0")"

./allclean.sh
exec python referenceFlow_setup.py
