#!/bin/bash
set -e

cd "$(dirname "$0")"

# Every case of the sensitivity study, one after another, each written to
# solution/sensitivity_analysis/<name>. A case resumes from its latest coupled backup; cases that
# are finished are commented out (# done), so running this script again skips them. Every case
# imports the solver of the campaign, the frozen export of commit 3f771630.
export PYTHONPATH=${STUDY_SOLVER:-$HOME/openonda_runs/study_2026-11/solver_3f771630}

# Baseline (to t = 100 s, the sampling floor) and N = 2
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --end-time 100 --name box2.4_sub5_inset8.75  # done
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --end-time 55 --name box2.4_sub2_inset8.75  # done

# k. Coupling iterations M
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --coupling-iterations 1 --end-time 55 --name box2.4_sub5_inset8.75_k1  # done
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --coupling-iterations 2 --coupling-tolerance 1e-30 --end-time 55 --name box2.4_sub5_inset8.75_k2  # done
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --coupling-iterations 4 --coupling-tolerance 1e-30 --end-time 55 --name box2.4_sub5_inset8.75_k4

# n_interval. Coupling interval N
python setup.py --box-margin 2.4 --fvm-substeps 1 --transfer-inset 8.75 --end-time 55 --name box2.4_sub1_inset8.75
# python setup.py --box-margin 2.4 --fvm-substeps 3 --transfer-inset 8.75 --end-time 55 --name box2.4_sub3_inset8.75  # done
python setup.py --box-margin 2.4 --fvm-substeps 6 --transfer-inset 8.75 --end-time 55 --name box2.4_sub6_inset8.75
python setup.py --box-margin 2.4 --fvm-substeps 1 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub1_inset8.75_kappa3
python setup.py --box-margin 2.4 --fvm-substeps 3 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub3_inset8.75_kappa3
python setup.py --box-margin 2.4 --fvm-substeps 4 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub4_inset8.75_kappa3
python setup.py --box-margin 2.4 --fvm-substeps 6 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub6_inset8.75_kappa3

# box_min. Smallest FVM boxes
# python setup.py --box-margin 1.6 --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 --name box1.6_sub5_inset8.75  # done
# python setup.py --box-margin 2.0 --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 --name box2.0_sub5_inset8.75  # done

# box_conv. Largest FVM boxes
# python setup.py --box-margin 3.2 --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 --name box3.2_sub5_inset8.75  # done
# python setup.py --box-margin 4.0 --fvm-substeps 5 --transfer-inset 8.75 --end-time 55 --name box4.0_sub5_inset8.75  # done

# inset. Transfer inset
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 4 --end-time 55 --name box2.4_sub5_inset4
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 14 --end-time 55 --name box2.4_sub5_inset14

# a. FVM cells at dt_FVM = 0.004 s, interface widths in 0.04 m cells
python setup.py --box-margin 2.4 --fvm-substeps 8 --transfer-inset 8.75 --cell-size 0.08 --fvm-time-step 0.004 --interface-cell-size 0.04 --end-time 55 --name box2.4_sub8_inset8.75_h0.08_dtf0.004_ic0.04
python setup.py --box-margin 2.4 --fvm-substeps 8 --transfer-inset 8.75 --cell-size 0.06 --fvm-time-step 0.004 --interface-cell-size 0.04 --end-time 55 --name box2.4_sub8_inset8.75_h0.06_dtf0.004_ic0.04
python setup.py --box-margin 2.4 --fvm-substeps 8 --transfer-inset 8.75 --fvm-time-step 0.004 --end-time 55 --name box2.4_sub8_inset8.75_dtf0.004

# a_fine. Finest FVM cells (about 5.8 GB of GPU memory)
python setup.py --box-margin 2.4 --fvm-substeps 8 --transfer-inset 8.75 --cell-size 0.03 --fvm-time-step 0.004 --interface-cell-size 0.04 --end-time 55 --name box2.4_sub8_inset8.75_h0.03_dtf0.004_ic0.04

# p. dt_FVM = 0.004 s at dt_VPM = 0.04 s
python setup.py --box-margin 2.4 --fvm-substeps 10 --transfer-inset 8.75 --fvm-time-step 0.004 --end-time 55 --name box2.4_sub10_inset8.75_dtf0.004

# c. dt_FVM = 0.002 s at dt_VPM = 0.04 s
python setup.py --box-margin 2.4 --fvm-substeps 20 --transfer-inset 8.75 --fvm-time-step 0.002 --end-time 55 --name box2.4_sub20_inset8.75_dtf0.002

# v_side. Particle side
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --particle-spacing 0.08 --end-time 55 --name box2.4_sub5_inset8.75_hp0.08
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --vpm-integrator rk4 --end-time 55 --name box2.4_sub5_inset8.75_rk4
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --vorticity-floor 0.001 --end-time 55 --name box2.4_sub5_inset8.75_vf0.001
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --vorticity-floor 0.1 --end-time 55 --name box2.4_sub5_inset8.75_vf0.1

# f_side. FVM side
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --fvm-outer-correctors 1 --end-time 55 --name box2.4_sub5_inset8.75_oc1
python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --fvm-outer-correctors 3 --end-time 55 --name box2.4_sub5_inset8.75_oc3
python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --fvm-time-step 0.02 --max-courant 0.5 --end-time 55 --name box2.4_sub2_inset8.75_dtf0.02_co0.5
python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --fvm-time-step 0.02 --max-courant 0.9 --end-time 55 --name box2.4_sub2_inset8.75_dtf0.02_co0.9

# b1. Strength-ratio limit kappa_max (postponed by the author)
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --max-strength-ratio 1.0 --end-time 55 --name box2.4_sub5_inset8.75_kappa1  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub5_inset8.75_kappa3  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --max-strength-ratio 2.0 --end-time 55 --name box2.4_sub5_inset8.75_kappa2  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --max-strength-ratio 10.0 --end-time 55 --name box2.4_sub5_inset8.75_kappa10  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --max-strength-ratio 1.0 --end-time 55 --name box2.4_sub2_inset8.75_kappa1  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --max-strength-ratio 3.0 --end-time 55 --name box2.4_sub2_inset8.75_kappa3  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --max-strength-ratio 2.0 --end-time 55 --name box2.4_sub2_inset8.75_kappa2  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --max-strength-ratio 10.0 --end-time 55 --name box2.4_sub2_inset8.75_kappa10  # postponed

# i_transfer. Transfer threshold and moment cap (postponed by the author)
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --transfer-removal-threshold 0.01 --end-time 55 --name box2.4_sub5_inset8.75_tau0.01  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --transfer-removal-threshold 0.2 --end-time 55 --name box2.4_sub5_inset8.75_tau0.2  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --moment-correction-fraction 0.02 --end-time 55 --name box2.4_sub5_inset8.75_mc0.02  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 5 --transfer-inset 8.75 --moment-correction-fraction 0.3 --end-time 55 --name box2.4_sub5_inset8.75_mc0.3  # postponed
# python setup.py --box-margin 2.4 --fvm-substeps 2 --transfer-inset 8.75 --transfer-removal-threshold 0.2 --end-time 55 --name box2.4_sub2_inset8.75_tau0.2  # postponed
