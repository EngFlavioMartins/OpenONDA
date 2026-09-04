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

# Direct wins for the O(10^3) fixed-population CS/RWM clouds. Treecode is used
# for DVH/GBD because their regenerated clouds grow beyond 100,000 particles;
# the production FMM currently does not support the Metal backend used here.

# Isolated vortex
python -u -m "${MODULE}.setup" vortex CS DIRECT
python -u -m "${MODULE}.setup" vortex DVH TREECODE
python -u -m "${MODULE}.setup" vortex GBD TREECODE
python -u -m "${MODULE}.assets.rwm_ensemble" vortex --number-of-realizations 10 --induction DIRECT

# Counter-rotating vortex pair
python -u -m "${MODULE}.setup" dipole CS DIRECT
python -u -m "${MODULE}.setup" dipole DVH TREECODE
python -u -m "${MODULE}.setup" dipole GBD TREECODE
python -u -m "${MODULE}.assets.rwm_ensemble" dipole --number-of-realizations 10 --induction DIRECT

# Co-rotating vortex pair
python -u -m "${MODULE}.setup" merging CS DIRECT
python -u -m "${MODULE}.setup" merging DVH TREECODE
python -u -m "${MODULE}.setup" merging GBD TREECODE
python -u -m "${MODULE}.assets.rwm_ensemble" merging --number-of-realizations 10 --induction DIRECT

# Aggregate RWM ensemble means and fields, then build all figures and the
# manifest from the samples, then run final independent validation.
python -m "${MODULE}.assets.postprocess" --aggregate-rwm \
  --expected-rwm-vortex-members 10 \
  --expected-rwm-dipole-members 10 \
  --expected-rwm-merging-members 10

tutorials/vpm/lamb_oseen_vortex/allplot.sh

python -m "${MODULE}.assets.postprocess"
