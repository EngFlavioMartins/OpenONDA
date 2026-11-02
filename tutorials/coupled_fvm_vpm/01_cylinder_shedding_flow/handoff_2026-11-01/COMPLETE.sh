#!/bin/bash
# Complete the coupled-cylinder sensitivity study in one solver version, unattended.
#
#   ./COMPLETE.sh              # phase 1 (new design), then phase 2 (earlier families rerun)
#   ./COMPLETE.sh --phase 1    # phase 1 only
#   ./COMPLETE.sh --with-fine  # also the runs that need more than 6 GB of GPU memory
#   ./COMPLETE.sh --dry-run    # show what is stored, what will run, and what is archived
#
# Restartable: finished runs are skipped, interrupted runs resume from their latest coupled
# backup. Three runs in a row that stop within 10 minutes halt the study (sensitivity/HALTED)
# instead of running through the list. Progress: sensitivity/COMPLETE.log.
#
# One solver version. Coupled runs stored before the interface fixes of 31 October 2026
# (direction-aware coupling condition, coupling-grade box mesh), or with the slab mesh before
# the z-face fix of 1 November 2026, are moved, not deleted, to
# sensitivity/archive_pre_2026-10-31/, so every compared run comes from the same code. The
# FVM-only references stay: those fixes do not touch a mesh without coupling patches.

set -u
cd "$(dirname "$0")"

PHASES="1 2"
WITH_FINE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASES=$2; shift ;;
        --with-fine) WITH_FINE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
    shift
done

# Phase 1: measurement, mechanism, domain, then the interface, particle and FVM sides.
#   h          baseline and N = 2 to 100 s: the sampling floor and the anchors of this version
#   g          FVM-only references with h cells over the coupled FVM domain
#   b1         strength limit kappa_max = 1, 2, 3, 10 at N = 2 and 5
#   n_interval N = 1 ... 6 at kappa_max = 1.8 and 3
#   e_old      N = 4 (N = 2 comes from h)
#   box_conv   FVM margins 3.2 and 4.0 D
#   i_transfer, v_side, f_side, e, d, a, c
PHASE_1=(h g b1 n_interval e_old box_conv i_transfer v_side f_side e d a c)
# Phase 2: the families of the 19-25 October study, rerun in this version.
PHASE_2=(baseline k c_ref x t2 p b f_fact r inset)

SECTIONS=()
for phase in $PHASES; do
    case $phase in
        1) SECTIONS+=("${PHASE_1[@]}") ;;
        2) SECTIONS+=("${PHASE_2[@]}") ;;
    esac
done
[[ $WITH_FINE -eq 1 ]] && SECTIONS+=(f)
export RUN_FINE=$WITH_FINE

mkdir -p sensitivity
LOG=sensitivity/COMPLETE.log
ARCHIVE=sensitivity/archive_pre_2026-10-31
FIX_TIME=$(date -d '2026-10-31 12:00' +%s)
SLAB_FIX_TIME=$(date -d '2026-11-01 10:30' +%s)

# Coupled runs with output from an older solver or mesher version.
stale_runs() {
    local directory forces stamp
    for directory in sensitivity/box*/; do
        directory=${directory%/}
        forces=$directory/samples/forces_history.csv
        [[ -f $forces ]] || continue
        stamp=$(stat -c %Y "$forces")
        if [[ $stamp -lt $FIX_TIME ]]; then
            echo "$directory"
        elif [[ $stamp -lt $SLAB_FIX_TIME ]] \
            && ! grep -q SLAB_HALF_DEPTH_LIMIT "$directory/setup.py" 2> /dev/null; then
            echo "$directory"
        fi
    done
}

if [[ $DRY_RUN -eq 1 ]]; then
    echo "sections in order: ${SECTIONS[*]}"
    echo
    echo "to be archived (older solver or mesher version):"
    stale_runs | sed 's/^/  /'
    echo
    echo "runs listed per section (finished ones are skipped when the study runs):"
    for section in "${SECTIONS[@]}"; do
        printf '  %-11s %s\n' "$section" \
            "$(sed -n "/^if want $section; then/,/^fi/p" assets/run_sensitivity_study.sh \
                | grep -cE '^\s*(GPU_MIB=\S+ |END_TIME=\S+ )?(run|reference) ')"
    done
    exit 0
fi

log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

rm -f sensitivity/HALTED sensitivity/.quick_failures
stale=$(stale_runs)
if [[ -n $stale ]]; then
    mkdir -p "$ARCHIVE"
    for directory in $stale; do
        log "archive $directory -> $ARCHIVE/"
        mv "$directory" "$ARCHIVE/"
    done
fi

log "start: sections ${SECTIONS[*]}"
for section in "${SECTIONS[@]}"; do
    log "section $section"
    bash assets/run_sensitivity_study.sh "$section" 2>&1 | tee -a "$LOG"
    if [[ ${PIPESTATUS[0]} -eq 3 || -f sensitivity/HALTED ]]; then
        log "HALTED in section $section at $(cat sensitivity/HALTED 2> /dev/null): see its run.log"
        exit 3
    fi
done

log "figures and tables"
python -m assets.window_uncertainty 2>&1 | tee -a "$LOG"
bash allplot_sensitivity.sh 2>&1 | tee -a "$LOG"
log "done"
touch sensitivity/COMPLETE
