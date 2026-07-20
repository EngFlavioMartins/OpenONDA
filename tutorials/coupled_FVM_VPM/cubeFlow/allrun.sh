#!/bin/sh
set -eu

cd "$(dirname "$0")" || exit 1

run_python() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "OpenONDA-VPM" ] || [ "${CONDA_DEFAULT_ENV:-}" = "OpenONDA" ]; then
        python "$@"
    elif command -v conda >/dev/null 2>&1; then
        for env in "${OPENONDA_CONDA_ENV:-OpenONDA-VPM}" OpenONDA-VPM OpenONDA; do
            if conda run -n "$env" python -c 'import sys' >/dev/null 2>&1; then
                conda run --no-capture-output -n "$env" python "$@"
                return
            fi
        done
        echo "cubeFlow requires a Conda environment from scripts/environment/environment.yml." >&2
        exit 1
    else
        echo "cubeFlow requires the OpenONDA Conda environment." >&2
        exit 1
    fi
}

./allclean.sh
run_python -u cube_setup.py
