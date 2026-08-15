#!/usr/bin/env bash
# Run the complete Lamb--Oseen benchmark (single vortex, dipole, merging
# pair, for every diffusion scheme), then make the figures.
# Usage: ./allrun.sh
set -uo pipefail

cd "$(dirname "$0")"

echo
echo "===== CLEAN ====="
echo
./allclean.sh

echo
echo "===== SIMULATE ====="
echo
status=0
rwm_realizations="${RWM_REALIZATIONS:-8}"

run_checked() {
    echo
    echo "---- $1 ----"
    shift
    if ! "$@"; then
        echo "WARNING: run failed; continuing so available data can still be plotted." >&2
        status=1
    fi
}

for case_spec in "vortex:+1:0" "dipole:+1:-1" "merging:+1:+1"; do
    IFS=: read -r case_name gamma1 gamma2 <<< "$case_spec"
    for scheme in cs dvh gbd; do
        run_checked "$case_name / $scheme" \
            python -u lambossen_setup.py \
                --gamma1 "$gamma1" --gamma2 "$gamma2" --schemes "$scheme"
    done
    run_checked "$case_name / RWM ${rwm_realizations}-member CPU ensemble" \
        python -u assets/run_rwm_ensemble.py \
            --gamma1 "$gamma1" --gamma2 "$gamma2" --realizations "$rwm_realizations"
done

echo
echo "===== FIGURES ====="
echo
if ! ./allplot.sh; then
    status=1
fi

echo
echo "===== DONE ====="
echo
exit "$status"
