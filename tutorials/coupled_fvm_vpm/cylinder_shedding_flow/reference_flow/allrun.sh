#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python}" -u setup.py --name coarse --dx 0.125
"${PYTHON:-python}" -u setup.py --name medium --dx 0.0625
"${PYTHON:-python}" -u setup.py --name fine --dx 0.03125
