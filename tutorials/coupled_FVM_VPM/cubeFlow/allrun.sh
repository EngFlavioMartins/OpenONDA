#!/bin/sh
set -eu

case_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$case_dir/../../.." && pwd)
cd "$case_dir"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

./allclean.sh
exec python cubeFlow_setup.py
