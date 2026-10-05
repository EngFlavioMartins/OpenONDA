#!/bin/bash
set -e

cd "$(dirname "$0")"

./allclean.sh

python -m openonda.tutorial_runner . setup
