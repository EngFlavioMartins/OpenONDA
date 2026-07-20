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
        echo "referenceFlow requires a Conda environment from scripts/environment/environment.yml." >&2
        exit 1
    else
        echo "referenceFlow requires the OpenONDA Conda environment." >&2
        exit 1
    fi
}

run_mpi_python() {
    ranks="$1"
    shift
    if [ "${CONDA_DEFAULT_ENV:-}" = "OpenONDA-VPM" ] || [ "${CONDA_DEFAULT_ENV:-}" = "OpenONDA" ]; then
        if ! command -v mpiexec >/dev/null 2>&1; then
            echo "OPENONDA_FVM_MPI_RANKS requires mpiexec in the active Conda environment." >&2
            exit 1
        fi
        mpiexec -n "$ranks" python "$@"
        return
    fi
    for env in "${OPENONDA_CONDA_ENV:-OpenONDA-VPM}" OpenONDA-VPM OpenONDA; do
        if conda run -n "$env" mpiexec --version >/dev/null 2>&1; then
            conda run --no-capture-output -n "$env" mpiexec -n "$ranks" python "$@"
            return
        fi
    done
    echo "referenceFlow requires a Conda environment from scripts/environment/environment.yml." >&2
    exit 1
}

./allclean.sh
ranks="${OPENONDA_FVM_MPI_RANKS:-1}"
case "$ranks" in
    ''|*[!0-9]*) echo "OPENONDA_FVM_MPI_RANKS must be a positive integer." >&2; exit 2 ;;
esac
if [ "$ranks" -gt 1 ]; then
    export OPENONDA_FVM_MPI_RANKS="$ranks"
    run_mpi_python "$ranks" -u reference_setup.py
else
    run_python -u reference_setup.py
fi
