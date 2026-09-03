#!/usr/bin/env bash
# Run the Lamb--Oseen vortex, dipole, and merging-vortex comparisons end to end.
# All output is written beneath this tutorial directory; the working directory
# is the repository root only so that the `tutorials` package is importable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.lamb_oseen_vortex"

# Clean the tutorial-local output directories first (never repository root).
tutorials/vpm/lamb_oseen_vortex/allclean.sh

# Isolated vortex
python -u -m "${MODULE}.setup" vortex CS
python -u -m "${MODULE}.setup" vortex DVH
python -u -m "${MODULE}.setup" vortex GBD
python -u -m "${MODULE}.assets.rwm_ensemble" vortex --number-of-realizations 10

# Counter-rotating vortex pair
python -u -m "${MODULE}.setup" dipole CS
python -u -m "${MODULE}.setup" dipole DVH
python -u -m "${MODULE}.setup" dipole GBD
python -u -m "${MODULE}.assets.rwm_ensemble" dipole --number-of-realizations 10

# Co-rotating vortex pair
python -u -m "${MODULE}.setup" merging CS
python -u -m "${MODULE}.setup" merging DVH
python -u -m "${MODULE}.setup" merging GBD
python -u -m "${MODULE}.assets.rwm_ensemble" merging --number-of-realizations 10

# Aggregate RWM ensemble means and fields, then build all figures and the
# manifest from the samples, then run final independent validation.
python -m "${MODULE}.assets.postprocess" --aggregate-rwm \
  --expected-rwm-vortex-members 8 \
  --expected-rwm-dipole-members 12 \
  --expected-rwm-merging-members 8

tutorials/vpm/lamb_oseen_vortex/allplot.sh

python -m "${MODULE}.assets.postprocess"
