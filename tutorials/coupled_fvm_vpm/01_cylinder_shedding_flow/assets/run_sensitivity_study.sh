#!/bin/bash
# Sensitivity study of the coupled cylinder. Run from the cylinder case directory:
#
#   assets/run_sensitivity_study.sh            # every section, one run after another
#   assets/run_sensitivity_study.sh b1 a       # the named sections only
#   assets/run_sensitivity_study.sh --list     # section names
#
# Each coupled run solves a copy of this case in sensitivity/box{margin}_sub{N}_inset{cells}{suffix},
# each FVM-only variant a copy of reference_flow/ in sensitivity/reference_{suffix}. The names below
# are the names of the stored runs, so a section that is launched again resumes its runs from their
# latest backups, and a run that already reached the end time is skipped. Suffix order (parsed by
# assets/plot_sensitivity.py): _xd _h _dtf _ic _hp _bs _be _kappa _tau _mc _rk _vf _co _oc _k _re _rep.
#
# Sections are grouped by what they isolate (see claude/cylinder_sensitivity_redesign_2026-10-31.md
# of the OpenONDA project): D the FVM domain, I the interface at a fixed FVM and VPM, T the coupling
# interval, V the particle side at a fixed FVM and interface, F the FVM side at a fixed interface
# and particle set. Every new run holds the baseline cells, steps and Reynolds number unless the
# section varies them.
#
# GPU memory. A coupled run at h = 0.04 m holds about 3 GB of particle images on the GPU; without
# free memory the image sums fall back to the CPU, about 50 times slower. Before each coupled run the
# script waits until MIN_FREE_GPU_MIB (default 3500) MiB are free, so it never starts next to another
# GPU job (the cube or wing tutorial, or a second study). Runs on finer particles need more:
# h = 0.0283 m about 6 GB, h_VPM = 0.02 m about 12 GB. Those sections raise the requirement
# themselves; on a 6 GB card they run only with the GPU otherwise idle, and section f does not fit.

set -u
cd "$(dirname "$0")/.."

END_TIME=${END_TIME:-55}               # s, end of every study run
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-3500}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-300}

# Wait until the GPU has the given free memory [MiB].
wait_for_gpu() {
    local needed=${1:-$MIN_FREE_GPU_MIB}
    command -v nvidia-smi > /dev/null || return 0
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
        [[ ${free:-0} -ge $needed ]] && return 0
        echo "$(date '+%F %T')  waiting for ${needed} MiB of free GPU memory (${free} MiB free)"
        sleep "$GPU_POLL_SECONDS"
    done
}

# Whether a run directory already holds forces up to the end time.
finished() {
    local forces="$1/samples/forces_history.csv"
    [[ -f $forces ]] || return 1
    python - "$forces" "$END_TIME" << 'EOF'
import sys
import numpy as np
time = np.atleast_1d(np.genfromtxt(sys.argv[1], delimiter=",", names=True)["time"])
sys.exit(0 if time.size and time[-1] >= float(sys.argv[2]) - 0.1 else 1)  # within a sample
EOF
}

# Coupled run. Arguments: FVM margin [D], FVM substeps, transfer inset [cells], name suffix,
# then the changed setup inputs. GPU_MIB overrides the free GPU memory the run waits for.
run() {
    local directory="sensitivity/box$1_sub$2_inset$3$4"
    if finished "$directory"; then
        echo "skip $directory (reached t = $END_TIME s)"
        return 0
    fi
    wait_for_gpu "${GPU_MIB:-$MIN_FREE_GPU_MIB}"
    echo "$(date '+%F %T')  run $directory"
    mkdir -p "$directory"
    cp -r --update=none setup.py assets "$directory"
    (cd "$directory" && python setup.py --box-margin "$1" --fvm-substeps "$2" --transfer-inset "$3" \
        "${@:5}" --end-time "$END_TIME") > "$directory/run.log" 2>&1 \
        || echo "$(date '+%F %T')  $directory stopped, see $directory/run.log"
}

# FVM-only variant on the CPU. Arguments: name suffix, then the changed setup inputs.
reference() {
    local directory="sensitivity/reference_$1"
    if finished "$directory"; then
        echo "skip $directory (reached t = $END_TIME s)"
        return 0
    fi
    echo "$(date '+%F %T')  reference $directory"
    mkdir -p "$directory"
    cp -r --update=none reference_flow/setup.py reference_flow/assets "$directory"
    (cd "$directory" && python setup.py "${@:2}" --end-time "$END_TIME") > "$directory/run.log" 2>&1 \
        || echo "$(date '+%F %T')  $directory stopped, see $directory/run.log"
}

SECTIONS=(h g b1 n_interval box_conv i_transfer v_side f_side d c a e baseline k c_ref x t2 p e_old b f_fact r inset f)
if [[ ${1:-} == --list ]]; then
    printf '%s\n' "${SECTIONS[@]}"
    exit 0
fi
selected=("$@")
[[ ${#selected[@]} -eq 0 ]] && selected=("${SECTIONS[@]}")
want() { [[ " ${selected[*]} " == *" $1 "* ]]; }

# ------------------------------------------------------------------------------------------------
# Sections added on 30 October 2026 (thesis section 4.7; follow-up of the error study)
# ------------------------------------------------------------------------------------------------

# b1. Strength-ratio limit kappa_max. In the N = 2 run kappa sits at kappa_max = 1.8 in 81% of the
# transfers, in the baseline never. kappa_max = 1 gives gamma = 0 (direct assignment of the
# mismatch); kappa_max = 3 leaves the limit inactive. Isolates the strength correction at the wall
# from the number of transfers per convective time.
# The gain is gamma = min(kappa_max - 1, 1): kappa_max = 2 is the smallest value with full gain, so
# kappa_max = 2, 3 and 10 separate the gain from the bound on the corrected strength.
if want b1; then
    run 2.4 2 8.75 _kappa1 --max-strength-ratio 1.0
    run 2.4 5 8.75 _kappa1 --max-strength-ratio 1.0
    run 2.4 2 8.75 _kappa3 --max-strength-ratio 3.0
    run 2.4 5 8.75 _kappa3 --max-strength-ratio 3.0
    run 2.4 2 8.75 _kappa2 --max-strength-ratio 2.0
    run 2.4 5 8.75 _kappa2 --max-strength-ratio 2.0
    run 2.4 2 8.75 _kappa10 --max-strength-ratio 10.0
    run 2.4 5 8.75 _kappa10 --max-strength-ratio 10.0
fi

# ------------------------------------------------------------------------------------------------
# Sections added on 31 October 2026 (redesign: isolate the interface from the FVM and the VPM)
# ------------------------------------------------------------------------------------------------

# T. Coupling interval at the baseline FVM step, N = 1 to 6, at the default and at an inactive
# strength limit. The stored interval series (N = 2, 4, 5 and the dt_FVM variants) gives a coupling
# error that grows as the interval shrinks and collapses on dt_VPM; this section decides whether
# the per-transfer error is the limiter (curves separate) or lies elsewhere (curves coincide).
# N = 6 is the largest admissible interval (Lambda = 0.93); N = 1 transfers every FVM step.
if want n_interval; then
    run 2.4 1 8.75 ""
    run 2.4 3 8.75 ""
    run 2.4 6 8.75 ""
    run 2.4 1 8.75 _kappa3 --max-strength-ratio 3.0
    run 2.4 3 8.75 _kappa3 --max-strength-ratio 3.0
    run 2.4 4 8.75 _kappa3 --max-strength-ratio 3.0
    run 2.4 6 8.75 _kappa3 --max-strength-ratio 3.0
fi

# D. FVM domain beyond the baseline margin. The lateral series 1.4, 1.6, 1.8, 2.4 D has not
# flattened at 2.4 D (lift-amplitude coupling error -5.4, -3.6, -2.0, -1.4 %); two wider boxes
# show where the domain error ends and whether the baseline carries any. The downstream face
# keeps the margin + 2 D rule. Mostly CPU: the FVM slab grows, the particle set hardly does.
if want box_conv; then
    run 3.2 5 8.75 ""
    run 4.0 5 8.75 ""
fi

# I. Transfer thresholding and conservation at a fixed FVM, particle set and interval. The removal
# threshold tau_transfer is at least the GBD floor (0.01 1/s); the moment-correction cap is in
# (0, 1]. Baseline 0.05 1/s and 0.08.
if want i_transfer; then
    run 2.4 5 8.75 _tau0.01 --transfer-removal-threshold 0.01
    run 2.4 5 8.75 _tau0.2 --transfer-removal-threshold 0.2
    run 2.4 5 8.75 _mc0.02 --moment-correction-fraction 0.02
    run 2.4 5 8.75 _mc0.3 --moment-correction-fraction 0.3
    run 2.4 2 8.75 _tau0.2 --transfer-removal-threshold 0.2    # at the short interval, where kappa saturates
fi

# V. Particle side at the baseline FVM and interface: particles twice as coarse as the cells (the
# cheapest run of the study; the span follows the particle spacing, so the FVM cell is extruded
# to 0.08 m), the integrator, and the removal floor of weak diffused particles.
if want v_side; then
    run 2.4 5 8.75 _hp0.08 --particle-spacing 0.08
    run 2.4 5 8.75 _rk4 --vpm-integrator rk4
    run 2.4 5 8.75 _vf0.001 --vorticity-floor 0.001
    run 2.4 5 8.75 _vf0.1 --vorticity-floor 0.1
fi

# F. FVM side at the baseline interface and particle set: PIMPLE outer correctors at the fixed
# step, and the Courant controller. With --max-courant the FVM step argument is the initial and
# maximum step and N the nominal ratio, so dt_VPM <= N dt_FVM,max = 0.04 s; the coupler shortens
# the interval with the FVM Courant step, so these two runs show what a user gets with the
# controller on rather than isolating one parameter.
if want f_side; then
    run 2.4 5 8.75 _oc1 --fvm-outer-correctors 1
    run 2.4 5 8.75 _oc3 --fvm-outer-correctors 3
    run 2.4 2 8.75 _dtf0.02_co0.5 --fvm-time-step 0.02 --max-courant 0.5
    run 2.4 2 8.75 _dtf0.02_co0.9 --fvm-time-step 0.02 --max-courant 0.9
fi

# d. Coupling interval at the baseline FVM step: two more levels, so that the trend with
# D/(U dt_VPM) does not rest on one run at 125 intervals per convective time. N = 6 gives
# Lambda = 0.93; N = 8 is inadmissible (Lambda = 1.24).
if want d; then
    run 2.4 3 8.75
    run 2.4 6 8.75
fi

# c. Temporal order at a fixed coupling interval dt_VPM = 0.04 s (N = 5 and 10 are stored).
if want c; then
    reference dt0.001 --time-step 0.001
    run 2.4 20 8.75 _dtf0.002 --fvm-time-step 0.002
    run 2.4 40 8.75 _dtf0.001 --fvm-time-step 0.001
fi

# a. Spatial order at a coupling interval admissible on the finest cells: dt_VPM = 0.032 s
# (N = 8 at dt_FVM = 0.004 s; Lambda = 0.83 with the 25.8 1/s strain of h = 0.0283 m), interface
# widths fixed in 0.04 m cells. The FVM-only references at dt_FVM = 0.004 s exist for every level.
if want a; then
    run 2.4 8 8.75 _h0.08_dtf0.004_ic0.04 --cell-size 0.08 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
    run 2.4 8 8.75 _h0.056569_dtf0.004_ic0.04 --cell-size 0.056569 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
    run 2.4 8 8.75 _dtf0.004 --fvm-time-step 0.004
    GPU_MIB=${GPU_MIB_FINE:-5800} run 2.4 8 8.75 _h0.028284_dtf0.004_ic0.04 --cell-size 0.028284 \
        --fvm-time-step 0.004 --interface-cell-size 0.04
fi

# e. Blending depths delta_0 (zero-weight layer) and delta_1 (start of the replacement region),
# in cells below the transfer surface; baseline 2 and 6.
if want e; then
    run 2.4 5 8.75 _be3 --blend-end 3
    run 2.4 5 8.75 _be10 --blend-end 10
    run 2.4 5 8.75 _bs0 --blend-start 0
    run 2.4 5 8.75 _bs4 --blend-start 4
fi

# g. FVM-only reference with the coupled cells over the whole coupled FVM domain, so that the
# coupling error no longer contains the coarser 2h cells of the reference between x = 1.5 and 4.4 m.
if want g; then
    reference fw4.4 --fine-wake-end 4.4
fi

# h. Statistics window: the baseline and the N = 2 run continued to t = 100 s, so that the
# 40-55 s window (2.7 periods) can be compared with 55-100 s.
if want h; then
    END_TIME=100 run 2.4 5 8.75 ""
    END_TIME=100 run 2.4 2 8.75 ""
fi

# ------------------------------------------------------------------------------------------------
# Sections of the error study of 19-25 October 2026 (stored; listed so that they reproduce)
# ------------------------------------------------------------------------------------------------

if want baseline; then
    run 2.4 5 8.75 ""
    run 2.4 5 8.75 _rep1          # repeat of the baseline: GPU summation order
fi

# K. Fixed number of coupling iterations (_k3 stopped on GPU memory; _k6 not run).
if want k; then
    run 2.4 5 8.75 _k1 --coupling-iterations 1
    run 2.4 5 8.75 _k2 --coupling-iterations 2 --coupling-tolerance 1e-30
    run 2.4 5 8.75 _k4 --coupling-iterations 4 --coupling-tolerance 1e-30
fi

# C. Cells, steps and particles refined together at constant Courant number, N = 2, with the
# references that give Q_ext.
if want c_ref; then
    reference h0.08 --cell-size 0.08
    reference h0.056569 --cell-size 0.056569
    reference h0.028284 --cell-size 0.028284
    run 2.4 2 8.75 _h0.08 --cell-size 0.08
    run 2.4 2 8.75 _h0.056569 --cell-size 0.056569
    run 2.4 2 8.75 ""
fi

# X. Cells refined at dt_FVM = 0.004 s, N = 4, interface widths in 0.04 m cells; and XP, the same
# at a fixed coupling interval of 0.04 s (N = 10).
if want x; then
    reference h0.08_dt0.004 --cell-size 0.08 --time-step 0.004
    reference h0.056569_dt0.004 --cell-size 0.056569 --time-step 0.004
    reference h0.028284_dt0.004 --cell-size 0.028284 --time-step 0.004
    reference dt0.004 --time-step 0.004
    run 2.4 4 8.75 _h0.08_dtf0.004_ic0.04 --cell-size 0.08 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
    run 2.4 4 8.75 _h0.056569_dtf0.004_ic0.04 --cell-size 0.056569 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
    run 2.4 4 8.75 _dtf0.004 --fvm-time-step 0.004
    run 2.4 10 8.75 _h0.08_dtf0.004_ic0.04 --cell-size 0.08 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
    run 2.4 10 8.75 _h0.056569_dtf0.004_ic0.04 --cell-size 0.056569 --fvm-time-step 0.004 \
        --interface-cell-size 0.04
fi

# T2. Both steps refined at N = 2 on the baseline cells.
if want t2; then
    reference dt0.016 --time-step 0.016
    reference dt0.002 --time-step 0.002
    run 2.4 2 8.75 _dtf0.016 --fvm-time-step 0.016
    run 2.4 2 8.75 _dtf0.004 --fvm-time-step 0.004
fi

# P. FVM step halved at the baseline coupling interval.
if want p; then
    run 2.4 10 8.75 _dtf0.004 --fvm-time-step 0.004
fi

# E. FVM substeps at the baseline FVM step (N = 3 and 6 are section d).
if want e_old; then
    run 2.4 2 8.75 ""
    run 2.4 4 8.75 ""
fi

# B. Lateral clearance (margin at a fixed downstream face of 4.4 D) and wake reach (downstream face
# at the baseline margin), at Re = 150 and 250.
if want b; then
    for margin in 1.4 1.6 1.8; do run "$margin" 5 8.75 _xd4.4 --box-downstream 4.4; done
    for downstream in 2.97 3.36; do run 2.4 5 8.75 "_xd$downstream" --box-downstream "$downstream"; done
    for margin in 1.36 1.52 1.68; do
        run "$margin" 5 8.75 _xd4.4_re250 --box-downstream 4.4 --reynolds-number 250
    done
    for downstream in 1.99 2.24 2.9; do
        run 2.4 5 8.75 "_xd${downstream}_re250" --box-downstream "$downstream" --reynolds-number 250
    done
fi

# F. 2x2 factorial of the downstream face (3.36 / 4.4 D) and N (2 / 5).
if want f_fact; then
    run 2.4 2 8.75 _xd3.36 --box-downstream 3.36
fi

# R. Reynolds number with the baseline rules.
if want r; then
    reference re100 --reynolds-number 100
    reference re250 --reynolds-number 250
    run 2.4 5 8.75 _re100 --reynolds-number 100
    run 2.4 2 8.75 _re100 --reynolds-number 100
    run 2.4 5 8.75 _re250 --reynolds-number 250
    run 2.4 4 8.75 _re250 --reynolds-number 250
    run 2.4 2 8.75 _re250 --reynolds-number 250
fi

# Transfer inset.
if want inset; then
    run 2.4 5 4 ""
    run 2.4 5 14 ""
fi

# f. Particle spacing h_VPM = h_FVM/2 (family S; not run). Four times the particles of the baseline:
# it needs a GPU with about 12 GB free and does not fit on a 6 GB card.
if want f; then
    GPU_MIB=${GPU_MIB_HP:-11500} run 2.4 5 8.75 _hp0.02 --particle-spacing 0.02
fi
