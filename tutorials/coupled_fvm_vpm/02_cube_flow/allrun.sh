#!/bin/bash -e
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TI_CPU_MAX_NUM_THREADS="${TI_CPU_MAX_NUM_THREADS:-4}"
exec python setup.py
