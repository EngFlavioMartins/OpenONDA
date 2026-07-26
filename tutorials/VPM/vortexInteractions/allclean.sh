#!/usr/bin/env bash
# Clean up solution files and figures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -eq 0 ]]; then
    rm -rf -- "solution" "figures"
    echo "Removed: solution/ figures/"
elif [[ $# -eq 1 && "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
    rm -rf -- "solution/${1}"
    echo "Removed: solution/${1}"
else
    echo "Usage: $0 [CASE_NAME]" >&2
    exit 2
fi
