#!/bin/bash
set -e

cd "$(dirname "$0")"

# The 23 coupled runs of the interface study, each written to solution/sensitivity_analysis/<name>
# and resumed from its latest coupled backup. All import the frozen export of commit 1e02b236, the
# conservative wall transfer of 11 Nov 2026 (the campaign of 2 to 8 Nov used 3f771630). The FVM-only
# references are in reference_flow/allrun.sh.
export PYTHONPATH=${STUDY_SOLVER:-$HOME/openonda_runs/solver_1e02b236}

run() { python setup.py --box-margin 2.4 --transfer-inset 8.75 --end-time 55 "$@"; }

# Recommended setup (the baseline): h = 0.04 m, dt_FVM = 0.008 s, N = 5 (dt_VPM = 0.04 s), to
# 100 s. The forces and profiles in figures/ come from this run.
run --fvm-substeps 5 --end-time 100 --name box2.4_sub5_inset8.75

# Coupling interval N, at two strength limits
for n in 1 2 3 6; do run --fvm-substeps $n --name box2.4_sub${n}_inset8.75; done
for n in 1 3 4 6; do run --fvm-substeps $n --max-strength-ratio 3.0 --name box2.4_sub${n}_inset8.75_kappa3; done

# Coupling iterations M (fixed count instead of the tolerance)
run --fvm-substeps 5 --coupling-iterations 1 --name box2.4_sub5_inset8.75_k1
run --fvm-substeps 5 --coupling-iterations 2 --coupling-tolerance 1e-30 --name box2.4_sub5_inset8.75_k2
run --fvm-substeps 5 --coupling-iterations 4 --coupling-tolerance 1e-30 --name box2.4_sub5_inset8.75_k4

# FVM box margin (upstream and sideways)
for margin in 1.6 2.0 3.2 4.0; do
    python setup.py --box-margin $margin --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 \
        --name box${margin}_sub5_inset8.75
done

# Transfer inset
for inset in 4 14; do
    python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset $inset --end-time 55 \
        --name box2.4_sub5_inset${inset}
done

# Cell size at a fixed coupling interval of 0.048 s (h = 0.04 m: box2.4_sub6_inset8.75)
run --fvm-substeps 3 --cell-size 0.08 --name box2.4_sub3_inset8.75_h0.08
run --fvm-substeps 4 --cell-size 0.06 --name box2.4_sub4_inset8.75_h0.06

# Particle side
run --fvm-substeps 5 --particle-spacing 0.08 --name box2.4_sub5_inset8.75_hp0.08
run --fvm-substeps 5 --vpm-integrator rk4 --name box2.4_sub5_inset8.75_rk4
run --fvm-substeps 5 --vorticity-floor 0.001 --name box2.4_sub5_inset8.75_vf0.001
