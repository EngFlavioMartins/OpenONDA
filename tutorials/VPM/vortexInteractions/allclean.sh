#!/usr/bin/env bash
# Delete one named case directory.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 CASE_NAME" >&2
    exit 2
fi

name="$1"
rm -rf -- "solution/${name}"
echo "Removed: solution/${name}"
