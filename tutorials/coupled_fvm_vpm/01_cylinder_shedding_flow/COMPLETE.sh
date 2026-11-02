#!/bin/bash
# Complete the coupled-cylinder sensitivity study in one solver version, unattended.
#
#   ./COMPLETE.sh              # phase 1 (new design), then phase 2 (earlier families rerun)
#   ./COMPLETE.sh --phase 1    # phase 1 only
#   ./COMPLETE.sh --with-fine  # also the runs that need more than 6 GB of GPU memory
#   ./COMPLETE.sh --dry-run    # show what is stored, what will run, and what is archived
#   ./COMPLETE.sh --slots 2    # two coupled runs at a time, references beside them on the CPU
#   ./COMPLETE.sh --sections "h box_conv n_interval"   # these sections only, in this order
#
# Restartable: finished runs are skipped, interrupted runs resume from their latest coupled
# backup. Three runs in a row that stop within 10 minutes halt the study (sensitivity/HALTED)
# instead of running through the list. Progress: sensitivity/COMPLETE.log.
#
# One solver version. Coupled runs stored before the coupling-grade box mesh of 31 October
# 2026, or with the slab mesh before the z-face fix of 1 November 2026, are moved, not
# deleted, to sensitivity/archive_pre_2026-10-31/; coupled runs made with solver 613b5898,
# whose direction-aware coupling condition lowered the cylinder drag by 4 % (removed on
# 1 November 2026), move to sensitivity/archive_613b5898/. So every compared run comes from
# the same code. The FVM-only references stay: none of this touches a mesh without coupling
# patches.
#
# Solver. STUDY_SOLVER=<directory holding an openonda/ package> makes every run import that copy
# instead of the working tree, so that edits to the repository during a campaign of one to two
# weeks cannot change the solver between runs (for example a `git archive <commit> openonda`
# export). Each run directory records the package it imported in solver.txt.
#
# Memory. On a desktop, start the study as its own systemd user service with OOMPolicy=continue
# (systemd-run --user --unit=<name> -p OOMPolicy=continue <launcher>): with the default policy an
# out-of-memory kill of any process in the unit it was started from stops the whole unit, which
# took every run of the study down on 1 November 2026. Runs also wait for available RAM before
# they start (MIN_FREE_RAM_MIB, REFERENCE_RAM_MIB in assets/run_sensitivity_study.sh).
#
# Slots. A coupled run is bound by one CPU core (the serial FVM) and holds about 2.8 GB of the
# GPU, so a 6 GB card takes two at a time. With --slots N, N workers walk the same section list,
# each pinned to its own six P-core threads, and a run directory is held by one worker at a time;
# a third worker runs the FVM-only references on the E-cores. A run waits for MIN_FREE_GPU_MIB
# of free GPU memory before it starts (default 3500 with one slot, 3200 with more).

set -u
cd "$(dirname "$0")"

PHASES="1 2"
WITH_FINE=0
DRY_RUN=0
SLOTS=1
ONLY_SECTIONS=""
STAGGER_SECONDS=${STAGGER_SECONDS:-900}  # s between the starts of the coupled workers
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASES=$2; shift ;;
        --with-fine) WITH_FINE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --slots) SLOTS=$2; shift ;;
        --sections) ONLY_SECTIONS=$2; shift ;;
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
#   box_min    FVM margins 1.6 and 2.0 D: the smallest box at the baseline interval
#   i_transfer, inset, k: the interface parameters to calibrate (threshold, moment cap, inset,
#              coupling iterations)
#   v_side, f_side, p: particle and FVM sides at a fixed interface (p: FVM step halved at the
#              baseline interval)
#   e, d, a
PHASE_1=(h g b1 n_interval e_old box_conv box_min i_transfer inset k v_side f_side p e d a)
# Phase 2: the other families of the 19-25 October study, rerun in this version, then the
# temporal series at a fixed interval (c, the most expensive section).
PHASE_2=(baseline c_ref x t2 b f_fact r c)

SECTIONS=()
if [[ -n $ONLY_SECTIONS ]]; then
    read -r -a SECTIONS <<< "$ONLY_SECTIONS"
else
    for phase in $PHASES; do
        case $phase in
            1) SECTIONS+=("${PHASE_1[@]}") ;;
            2) SECTIONS+=("${PHASE_2[@]}") ;;
        esac
    done
fi
[[ $WITH_FINE -eq 1 ]] && SECTIONS+=(f)
export RUN_FINE=$WITH_FINE

mkdir -p sensitivity
LOG=sensitivity/COMPLETE.log
if [[ -n ${STUDY_SOLVER:-} ]]; then
    [[ -f $STUDY_SOLVER/openonda/__init__.py ]] || { echo "no openonda package in $STUDY_SOLVER"; exit 1; }
    export PYTHONPATH=$STUDY_SOLVER${PYTHONPATH:+:$PYTHONPATH}
fi
ARCHIVE=sensitivity/archive_pre_2026-10-31
ARCHIVE_613b5898=sensitivity/archive_613b5898
FIX_TIME=$(date -d '2026-10-31 12:00' +%s)
SLAB_FIX_TIME=$(date -d '2026-11-01 10:30' +%s)

# Coupled runs with output from an older solver or mesher version: "<directory> <archive>".
stale_runs() {
    local directory forces stamp
    for directory in sensitivity/box*/; do
        directory=${directory%/}
        forces=$directory/samples/forces_history.csv
        [[ -f $forces ]] || continue
        stamp=$(stat -c %Y "$forces")
        if [[ $stamp -lt $FIX_TIME ]]; then
            echo "$directory $ARCHIVE"
        elif [[ $stamp -lt $SLAB_FIX_TIME ]] \
            && ! grep -q SLAB_HALF_DEPTH_LIMIT "$directory/setup.py" 2> /dev/null; then
            echo "$directory $ARCHIVE"
        elif grep -q solver_613b5898 "$directory/solver.txt" 2> /dev/null; then
            echo "$directory $ARCHIVE_613b5898"
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

log() { echo "$(date '+%F %T')  ${WORKER:+[$WORKER] }$*" | tee -a "$LOG"; }

rm -f sensitivity/HALTED sensitivity/.quick_failures
while read -r directory archive; do
    [[ -n $directory ]] || continue
    mkdir -p "$archive"
    log "archive $directory -> $archive/"
    mv "$directory" "$archive/"
done <<< "$(stale_runs)"

# One worker: every section in order, until done or halted.
work() {
    for section in "${SECTIONS[@]}"; do
        [[ -f sensitivity/HALTED ]] && return 3
        log "section $section"
        bash assets/run_sensitivity_study.sh "$section" 2>&1 \
            | sed -u "s/^/${WORKER:+[$WORKER] }/" | tee -a "$LOG"
        if [[ ${PIPESTATUS[0]} -eq 3 || -f sensitivity/HALTED ]]; then
            log "HALTED in section $section at $(cat sensitivity/HALTED 2> /dev/null): see its run.log"
            return 3
        fi
    done
}

log "start: sections ${SECTIONS[*]}, $SLOTS slot(s), solver $(python -c \
    "import importlib.util; print(importlib.util.find_spec('openonda').origin)")"
if [[ $SLOTS -le 1 ]]; then
    work || exit 3
else
    export MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-3200}
    pids=()
    for ((slot = 0; slot < SLOTS; slot++)); do
        (
            taskset -cp "$((6 * slot))-$((6 * slot + 5))" $BASHPID > /dev/null
            sleep $((slot * STAGGER_SECONDS))
            export WORKER=gpu$slot STUDY_ROLE=gpu
            work
        ) &
        pids+=($!)
    done
    # The references start after the coupled workers, so that their meshers (up to 6 GB) do not
    # peak together on a 16 GB machine.
    (
        taskset -cp 12-19 $BASHPID > /dev/null
        sleep $((SLOTS * STAGGER_SECONDS))
        export WORKER=cpu STUDY_ROLE=cpu
        work
    ) &
    pids+=($!)
    halted=0
    for pid in "${pids[@]}"; do wait "$pid" || halted=1; done
    [[ $halted -eq 1 ]] && exit 3
fi

log "figures and tables"
python -m assets.window_uncertainty 2>&1 | tee -a "$LOG"
bash allplot_sensitivity.sh 2>&1 | tee -a "$LOG"
log "done"
touch sensitivity/COMPLETE
