#!/bin/bash -e

python assets/postprocess.py
python assets/plot_cylinder.py
python assets/compare_reference.py --reference "${REFERENCE_GRID:-medium}" --if-available
