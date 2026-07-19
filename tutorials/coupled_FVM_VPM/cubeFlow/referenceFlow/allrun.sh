#!/bin/sh
set -eu

cd "$(dirname "$0")" || exit 1

run_python() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "OpenONDA-VPM" ]; then
        python "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda run --no-capture-output -n OpenONDA-VPM python "$@"
    else
        echo "referenceFlow requires the OpenONDA-VPM Conda environment." >&2
        exit 1
    fi
}

./allclean.sh
run_python -u reference_setup.py
