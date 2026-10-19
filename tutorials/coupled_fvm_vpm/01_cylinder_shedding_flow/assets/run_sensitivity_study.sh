#!/bin/bash
cd "$(dirname "$0")/.." || exit  # the case directory

# Each run solves a copy of this case in sensitivity/<run>/ up to t = 55 s, where
# the lift-amplitude error has reached its developed level, and resumes its latest
# backup when launched again. A run that stops, such as one whose particle step
# exceeds the stability limit, does not stop the study. The arguments are the FVM
# box margin [D], the FVM steps per exchange, the transfer-region inset [cells],
# then a name suffix for each parameter that differs from the baseline, followed
# by its setup.py flags.
run() {
    directory="sensitivity/box$1_sub$2_inset$3$4"
    mkdir -p "$directory"
    cp -r --update=none setup.py assets "$directory"
    (cd "$directory" && python setup.py --box-margin "$1" --fvm-substeps "$2" --transfer-inset "$3" "${@:5}" --end-time 55)
}

# Each reference variant solves a copy of the reference flow in sensitivity/reference_<variant>/
reference() {
    directory="sensitivity/reference_$1"
    mkdir -p "$directory"
    cp -r --update=none reference_flow/setup.py reference_flow/assets "$directory"
    (cd "$directory" && python setup.py "${@:2}" --end-time 55)
}

# The first argument names one study section; without it every section runs in turn
study=${1:-all}

# Convergence: cells, steps and particles refined together by sqrt(2) at constant Courant
# number and two FVM steps per exchange, each level with a reference on the same cells
if [[ $study == all || $study == convergence ]]; then
    reference h0.08 --cell-size 0.08
    reference h0.0566 --cell-size 0.056569
    reference h0.0283 --cell-size 0.028284
    run 2.4 2 8.75 _h0.08 --cell-size 0.08
    run 2.4 2 8.75 _h0.0566 --cell-size 0.056569
    run 2.4 2 8.75
    run 2.4 2 8.75 _h0.0283 --cell-size 0.028284
fi

# S4: interface sweeps per exchange, exactly K with an unreachable tolerance
if [[ $study == all || $study == s4 ]]; then
    run 2.4 5 8.75 _k1 --interface-iterations 1
    run 2.4 5 8.75 _k2 --interface-iterations 2 --interface-tolerance 1e-30
    run 2.4 5 8.75 _k3 --interface-iterations 3 --interface-tolerance 1e-30
    run 2.4 5 8.75 _k4 --interface-iterations 4 --interface-tolerance 1e-30
    run 2.4 5 8.75 _k6 --interface-iterations 6 --interface-tolerance 1e-30
fi

# S1b: particle step fixed at 0.04 s, smaller FVM steps, with references at the same FVM steps
if [[ $study == all || $study == s1b ]]; then
    reference dt0.004 --time-step 0.004
    reference dt0.002 --time-step 0.002
    run 2.4 10 8.75 _dtf0.004 --fvm-time-step 0.004
    run 2.4 20 8.75 _dtf0.002 --fvm-time-step 0.002
    run 2.4 40 8.75 _dtf0.001 --fvm-time-step 0.001
fi

# S2: particle spacing of half a cell, FVM step fixed at 0.008 s
if [[ $study == all || $study == s2 ]]; then
    run 2.4 2 8.75 _hp0.02 --particle-spacing 0.02
    run 2.4 4 8.75 _hp0.02 --particle-spacing 0.02
    run 2.4 5 8.75 _hp0.02 --particle-spacing 0.02
    run 2.4 6 8.75 _hp0.02 --particle-spacing 0.02
fi

# S3: other Reynolds numbers at the baseline parameters, each with its own reference
if [[ $study == all || $study == s3 ]]; then
    reference re100 --reynolds-number 100
    reference re250 --reynolds-number 250
    run 2.4 5 8.75 _re100 --reynolds-number 100
    run 2.4 5 8.75 _re250 --reynolds-number 250
fi

# Domain: FVM box margin and transfer-region inset
if [[ $study == all || $study == domain ]]; then
    run 1.0 5 8.75
    run 1.4 5 8.75
    run 1.8 5 8.75
    run 2.2 5 8.75
    run 2.6 5 8.75
    run 2.4 5 4
    run 2.4 5 6
    run 2.4 5 14
    run 2.4 5 20
fi

# S1a: baseline and FVM steps per exchange at a fixed FVM step of 0.008 s
if [[ $study == all || $study == s1a ]]; then
    run 2.4 5 8.75
    run 2.4 2 8.75
    run 2.4 4 8.75
    run 2.4 6 8.75
    run 2.4 8 8.75
fi
