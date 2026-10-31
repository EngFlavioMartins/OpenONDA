#!/bin/bash
# Complete the coupled-cylinder sensitivity study: run every section not yet stored, in the
# priority order of claude/cylinder_sensitivity_redesign_2026-10-31.md, then redraw the figures.
#
#   ./COMPLETE.sh              # everything that fits a 6 GB GPU, one run after another
#   ./COMPLETE.sh --with-fine  # also section f (h_VPM = 0.02 m, needs about 12 GB)
#   ./COMPLETE.sh --dry-run    # list what would run
#
# assets/run_sensitivity_study.sh skips runs that reached the end time and resumes the others
# from their latest backup, so this script may be stopped and started again at any time. It
# waits for free GPU memory before each coupled run (MIN_FREE_GPU_MIB, default 3500).
# Progress is appended to sensitivity/COMPLETE.log.

set -u
cd "$(dirname "$0")"

WITH_FINE=0
DRY_RUN=0
for argument in "$@"; do
    case $argument in
        --with-fine) WITH_FINE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument $argument"; exit 1 ;;
    esac
done

# 1. Measurement: the 100 s runs that give the sampling floor, and the fine-wake references
#    that remove the near-wake mesh difference from the coupling error.
# 2. Mechanism of the dominant error: the strength limit and the coupling interval.
# 3. The baseline domain.
# 4. The interface details, and the particle and FVM sides at a fixed interface.
# 5. Remaining sections of the 30 October list (blending depths, orders).
SECTIONS=(
    h g
    b1 n_interval
    box_conv
    i_transfer v_side f_side
    e d a c
)
[[ $WITH_FINE -eq 1 ]] && SECTIONS+=(f)

mkdir -p sensitivity
LOG=sensitivity/COMPLETE.log
log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

if [[ $DRY_RUN -eq 1 ]]; then
    echo "sections in order: ${SECTIONS[*]}"
    echo "stored runs:"
    for directory in sensitivity/box* sensitivity/reference_*; do
        [[ -d $directory ]] || continue
        forces="$directory/samples/forces_history.csv"
        if [[ -f $forces ]]; then
            echo "  $(tail -n 1 "$forces" | cut -d, -f1 | xargs printf '%7.2f s')  $directory"
        else
            echo "    (no forces)  $directory"
        fi
    done
    exit 0
fi

log "start: sections ${SECTIONS[*]}"
for section in "${SECTIONS[@]}"; do
    log "section $section"
    assets/run_sensitivity_study.sh "$section" 2>&1 | tee -a "$LOG"
done

# The reference with the fine wake also at the halved step (family X / P reference).
log "section g at dt = 0.004 s"
(
    directory=sensitivity/reference_dt0.004_fw4.4
    if [[ -f $directory/samples/forces_history.csv ]] \
        && python - "$directory/samples/forces_history.csv" << 'EOF'
import sys
import numpy as np
time = np.atleast_1d(np.genfromtxt(sys.argv[1], delimiter=",", names=True)["time"])
sys.exit(0 if time.size and time[-1] >= 54.9 else 1)
EOF
    then
        log "skip $directory"
    else
        mkdir -p "$directory"
        cp -r --update=none reference_flow/setup.py reference_flow/assets "$directory"
        (cd "$directory" && python setup.py --time-step 0.004 --fine-wake-end 4.4 --end-time 55) \
            > "$directory/run.log" 2>&1 || log "$directory stopped, see $directory/run.log"
    fi
)

log "figures and tables"
python -m assets.window_uncertainty 2>&1 | tee -a "$LOG"
./allplot_sensitivity.sh 2>&1 | tee -a "$LOG"
log "done"
