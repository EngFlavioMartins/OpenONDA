#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark, validate it, and make its figures.

set -euo pipefail
cd "$(dirname "$0")"

./allclean.sh

SCHEMES=(cs rwm dvh gbd)

run() {
    echo
    echo "=== $1 ==="
    python -u vortex_setup.py "$@"
}

for scheme in "${SCHEMES[@]}"; do
    run "vortex_${scheme}" +1
    run "dipole_${scheme}" +1 -1
    run "merging_${scheme}" +1 +1
done

# A blown-up case still exits 0, so check the results before plotting them.
echo
echo "=== validation ==="
validation_status=0
for physics in vortex dipole merging; do
    for scheme in "${SCHEMES[@]}"; do
        python assets/validate_results.py "$physics" "$scheme" || validation_status=1
    done
done

./allplot.sh -png

if ((validation_status)); then
    echo "One or more cases failed validation; the figures above show the runs as computed." >&2
    exit 1
fi
