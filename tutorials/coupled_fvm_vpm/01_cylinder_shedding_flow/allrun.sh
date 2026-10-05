#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# A supervising flock process retains the lock even if MPI closes inherited FDs.
# Keep this file outside archived output directories and never unlink it.
case_directory="$(pwd -P)"
if [[ "${_OPENONDA_CASE_LOCK:-}" != "$case_directory" ]]; then
    status=0
    flock --nonblock --conflict-exit-code 75 --close "$case_directory/.openonda-run.lock" \
        env _OPENONDA_CASE_LOCK="$case_directory" bash "$case_directory/allrun.sh" "$@" || status=$?
    if [[ "$status" -eq 75 ]]; then
        printf 'OpenONDA case is already running: %s\n' "$case_directory" >&2
    fi
    exit "$status"
fi
unset _OPENONDA_CASE_LOCK

# Archive before Python relaunches itself under MPI, so this runs exactly once.
if [[ "${1:-}" == "--fresh" ]]; then
    python assets/prepare_fresh_run.py
    shift
fi

# CUDA headers on distribution-installed toolkits may need an explicit prefix.
if [[ -z "${CUDA_PATH:-}" && -f /usr/include/cuda_runtime.h ]]; then
    export CUDA_PATH=/usr
fi

# Without --fresh, keep results and resume the current native checkpoint.
exec python setup.py "$@"
