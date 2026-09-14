#!/bin/bash -e

cd -- "$(dirname -- "$0")"
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--keep-mesh" ]; }; then
    echo "Usage: $0 [--keep-mesh]" >&2
    exit 2
fi

rm -rf solution samples figures
if [ "${1:-}" != "--keep-mesh" ]; then
    rm -rf constant
fi
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log
