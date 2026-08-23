#!/bin/sh
set -eu

cd "$(dirname "$0")"

./allclean.sh
exec python reference_flow_setup.py
