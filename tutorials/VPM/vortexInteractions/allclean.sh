#!/usr/bin/env bash
# Explicit cleanup for vortexInteractions.
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
