#!/usr/bin/env bash
# Delete one explicitly named isolated run. Legacy solution/ data is never touched.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: $0 RUN_ID" >&2
    exit 2
fi

run_id="$1"
rm -rf -- "solution/runs/${run_id}" "figures/runs/${run_id}"
echo "Removed isolated run: ${run_id}"
