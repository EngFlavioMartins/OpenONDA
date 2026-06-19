#!/usr/bin/env bash
PYTHON="$(conda run -n OpenONDA-VPM which python 2>/dev/null || command -v python3 || command -v python)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./allclean.sh

"$PYTHON" assets/quad_setup.py

./allplot.sh
echo "All runs and plots complete."
