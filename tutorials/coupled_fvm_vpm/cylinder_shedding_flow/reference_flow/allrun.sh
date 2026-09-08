#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

printf '%s\n' 'Cleaning previous reference-flow results...'
./allclean.sh
mkdir -p solution samples

"${OPENONDA_PYTHON:-${PYTHON:-python}}" -u setup.py --name coarse --dx 0.125
"${OPENONDA_PYTHON:-${PYTHON:-python}}" -u setup.py --name medium --dx 0.0625
"${OPENONDA_PYTHON:-${PYTHON:-python}}" -u setup.py --name fine --dx 0.03125
"${OPENONDA_PYTHON:-${PYTHON:-python}}" -u setup.py --name very_fine --dx 0.01563