#!/usr/bin/env bash
# Full validation of the cylinder-shedding seed-amplitude hypothesis.
#
# 1. clean both cases
# 2. run the fully meshed FVM reference to saturation
# 3. run the coupled hybrid to saturation
# 4. numerical-integrity gate (incl. open-vortex-line leakage, lattice match)
# 5. quantitative Von-Karman instability analysis and verdict
# 6. all figures
# 7. archive this run (outputs + analysis + figures) under runs/<seed>_<ts>/
#
# Usage:
#   ./allvalidate.sh                    unseeded
#   OPENONDA_SEED_AMPLITUDE=1e-4 ./allvalidate.sh
#   OPENONDA_SMOKE=1 ./allvalidate.sh   short smoke (no acceptance verdict)
#
# Exit codes:
#   0  everything passed
#   1  any simulation/integrity/analysis step failed
set -euo pipefail

cd "$(dirname "$0")"

seed="${OPENONDA_SEED_AMPLITUDE:-0}"
smoke="${OPENONDA_SMOKE:-0}"

echo
echo "===== VALIDATE (seed=$seed, smoke=$smoke) ====="
echo
./allclean.sh

mkdir -p solution runs figures
mkdir -p reference_flow/solution

echo
echo "===== REFERENCE (fully meshed FVM) ====="
echo
(
    cd reference_flow
    ./allclean.sh
    mkdir -p solution
    python -u reference_flow_setup.py 2>&1 | tee solution/reference_flow.log
)

echo
echo "===== HYBRID (coupled FVM-VPM) ====="
echo
python -u cylinder_shedding_flow_setup.py 2>&1 | tee solution/cylinder_shedding_flow.log

echo
echo "===== INTEGRITY ====="
echo
python assets/check_run.py

echo
echo "===== INSTABILITY ANALYSIS ====="
echo
python assets/analyse_von_karman.py

if [ "$smoke" = "1" ]; then
    echo
    echo "===== SMOKE RUN: figures skipped, no acceptance verdict ====="
    echo
    archive_label="smoke"
else
    echo
    echo "===== FIGURES ====="
    echo
    ./allplot.sh png
    archive_label="full"
fi

echo
echo "===== ARCHIVE ====="
echo
stamp="$(date +%Y%m%dT%H%M%S)"
archive="runs/seed${seed}_${archive_label}_${stamp}"
mkdir -p "$archive"
cp -R solution "$archive/solution"
cp -R reference_flow/solution "$archive/reference_flow_solution" 2>/dev/null || true
cp -R samples "$archive/samples" 2>/dev/null || true
cp -R reference_flow/samples "$archive/reference_flow_samples" 2>/dev/null || true
cp -R figures "$archive/figures" 2>/dev/null || true
cp README.md "$archive/README.md"
echo "Archived run to: $archive"

echo
echo "===== VALIDATE DONE ====="
echo
if [ "$smoke" = "1" ]; then
    echo "Smoke run complete. No acceptance verdict for short runs."
else
    echo "Validation complete. Verdict is in the INSTABILITY ANALYSIS section above"
    echo "and in solution/analysis_summary.json."
fi