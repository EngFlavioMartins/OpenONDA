#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${OPENONDA_PYTHON:-python}" setup.py
