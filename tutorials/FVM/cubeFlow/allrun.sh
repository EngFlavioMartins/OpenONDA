#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

python assets/mesh_cube.py --mesh-size 20 --output assets/cubeFlow.msh
python cubeFlow_setup.py --end-time 30.0 --Re 300 --max-cfl 1.0 --write-interval-time 2.0
./allplot.sh
echo
echo "All runs and plots complete."
