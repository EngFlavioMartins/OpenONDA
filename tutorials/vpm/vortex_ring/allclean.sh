#!/usr/bin/env bash
# Remove only this tutorial's generated outputs. Called explicitly by allrun --clean.
set -euo pipefail

cd "$(dirname "$0")"
rm -rf solution samples figures
rm -f ./*.log ./run_manifest.json
