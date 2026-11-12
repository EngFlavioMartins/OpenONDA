#!/bin/bash
set -e

cd "$(dirname "$0")"

# Calibration study of the coupling interface: 19 coupled runs, each written to
# solution/sensitivity_analysis/<name> and resumed from its latest coupled backup. Every run holds
# the baseline below and changes one group's setting: h = 0.04 m cells and particles,
# dt_FVM = 0.008 s, FVM box margin 2.4 D, transfer inset 8.75 cells, coupling iterations to the
# tolerance 1e-5 from a linear first estimate, to t = 55 s. Each run is compared with the FVM-only
# reference on the same cells and step (reference_flow/allrun.sh).
#
# Names: box<margin>_sub<N>_inset<cells>, then one suffix per changed setting: _h<cell size>,
# _hp<particle spacing>, _k<coupling iterations>, _tol<tolerance>, _const (constant first estimate).

run() { python setup.py --box-margin 2.4 --transfer-inset 8.75 --end-time 55 "$@"; }

# Baseline, N = 5 (dt_VPM = 0.04 s), to 100 s: its 40-55 s window against its later whole
# periods measures the uncertainty of the statistics window of every other run.
run --fvm-substeps 5 --end-time 100 --name box2.4_sub5_inset8.75

# T. Coupling interval dt_VPM = N dt_FVM at the baseline FVM step: the error that each transfer
# adds, from a transfer every FVM step (N = 1) to the interval that the particle strain admits.
# The coupler shortens an interval whose particle strain would exceed its limit, so N = 10 also
# shows where that limit takes over.
for n in 1 2 3 6 10; do run --fvm-substeps $n --name box2.4_sub${n}_inset8.75; done

# M. Coupling iterations at N = 5: one pass, the tolerances 1e-3 and 1e-4 from the linear first
# estimate, and the tolerance 1e-5 from the constant first estimate as the converged-coupling
# limit. The baseline is the tolerance 1e-5 from the linear estimate.
run --fvm-substeps 5 --coupling-iterations 1 --name box2.4_sub5_inset8.75_k1
run --fvm-substeps 5 --coupling-tolerance 1e-3 --name box2.4_sub5_inset8.75_tol1e-3
run --fvm-substeps 5 --coupling-tolerance 1e-4 --name box2.4_sub5_inset8.75_tol1e-4
run --fvm-substeps 5 --first-estimate constant --name box2.4_sub5_inset8.75_const

# D. FVM box margin upstream and sideways (the downstream face 2 D farther): how far the particle
# boundary data can come from the body.
for margin in 1.6 2.0 3.2 4.0; do
    python setup.py --box-margin $margin --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 \
        --name box${margin}_sub5_inset8.75
done

# I. Transfer inset, the distance in cells from the outer FVM boundary to the transfer surface, in
# a fixed box: how close to the boundary the FVM vorticity may be handed to the particles.
for inset in 4 14; do
    python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset $inset --end-time 55 \
        --name box2.4_sub5_inset${inset}
done

# S. Spatial resolution at the fixed coupling interval 0.048 s: cells of 0.08 m (N = 3) and 0.06 m
# (N = 4), the FVM step, particle spacing and interface widths scaled with the cells, against
# h = 0.04 m at N = 6 (group T); then particles twice as coarse as the baseline cells.
run --fvm-substeps 3 --cell-size 0.08 --name box2.4_sub3_inset8.75_h0.08
run --fvm-substeps 4 --cell-size 0.06 --name box2.4_sub4_inset8.75_h0.06
run --fvm-substeps 5 --particle-spacing 0.08 --name box2.4_sub5_inset8.75_hp0.08
