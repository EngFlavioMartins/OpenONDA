#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# The supervisor owns the lock throughout Python/MPI, including fresh-run archiving.
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

if [[ "${1:-}" == "--fresh" ]]; then
    python ../assets/prepare_fresh_run.py --reference
    shift
fi

# One selected mesh. Preserve completed results and resume native backups.
exec python setup.py -h 0.04 "$@"
