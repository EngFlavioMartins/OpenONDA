#!/bin/bash

# Remove previous output and simulate from t = 0.
./allclean.sh
python setup.py "$@"
