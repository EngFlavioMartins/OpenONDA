#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

./allclean.sh


python setup.py
