#!/usr/bin/env bash
# Explicit cleanup for vortex_interactions.
#
# Unlike the single-case tutorial, this 6-case campaign runs for many minutes
# each, so a bare `./allclean.sh` refuses to wipe everything by accident.
# Clean one named case with `./allclean.sh CASE_NAME` or the whole set with
# `./allclean.sh --all`. solution/ samples/ figures/ held results are never
# overwritten by re-running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -eq 1 && "$1" == "--all" ]]; then
    rm -rf -- "solution" "samples" "figures"
    echo "Removed: solution/ samples/ figures/"
elif [[ $# -eq 1 && "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
    rm -rf -- "solution/${1}" "samples/${1}"
    echo "Removed: solution/${1} samples/${1}"
else
    echo "Refusing an implicit full cleanup." >&2
    echo "Usage: $0 CASE_NAME | $0 --all" >&2
    exit 2
fi
