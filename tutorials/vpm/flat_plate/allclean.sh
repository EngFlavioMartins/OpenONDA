#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -eq 1 && "$1" == "--all" ]]; then
    rm -rf -- "solution" "samples" "figures"
    rm -f -- ./*.log
    echo "Removed: solution/ samples/ figures/ *.log"
else
    echo "Refusing an implicit full cleanup." >&2
    echo "Usage: $0 --all" >&2
    exit 2
fi
