# Leapfrogging kinematic reference

`leapfrogging_lbm_trajectory.csv` contains the two vortex-core trajectories
shown for `Re = 3000` in Fig. 5(b) of:

M. Cheng, J. Lou, and T. T. Lim, "Leapfrogging of multiple coaxial viscous
vortex rings," *Physics of Fluids* 27, 031702 (2015),
https://doi.org/10.1063/1.4915890.

The marker centres were recovered from the vector paths in the published PDF.
The axes were calibrated from the vector tick positions at `Z/R0 = 2, 4, 6,
8` and `R/R0 = 0.6, 0.8, 1.0, 1.2`. The paper's axial coordinate `Z` is the
simulation's `x`, so the CSV stores it as `x_over_R0`. Cheng et al. initialize
the cores at `Z/R0 = 2` and `3`; the plot subtracts their midpoint, `2.5`, to
align those positions with the present centres at `x/R0 = -0.5` and `0.5`.
Only the arbitrary axial origin is changed.

The source case has `Re_Gamma = 3000`, `a0/R0 = 0.1`, and `h0/R0 = 1`, which
match the present leapfrogging geometry. The present initialization also uses
the single disturbance reported for the LBM comparison: amplitude
`epsilon/R0 = 0.05` and azimuthal mode `n = 8`. The disturbance phase is
arbitrary under rotation and is fixed to zero in `interactions_setup.py`.
