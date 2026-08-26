#!/usr/bin/env bash
# Remove generated solution, samples, and figure directories.
# Usage: ./allclean.sh --all
set -euo pipefail

cd "$(dirname "$0")"

if [ "${1:-}" != "--all" ]; then
    echo "Usage: $0 --all" >&2
    echo "Refusing to remove files without --all flag." >&2
    exit 1
fi

rm -rf solution samples figures
rm -f ./*.log
