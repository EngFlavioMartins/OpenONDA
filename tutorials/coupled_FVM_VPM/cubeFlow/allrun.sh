#!/bin/sh
set -eu

cd "$(dirname "$0")"

./allclean.sh
exec python cubeFlow_setup.py
