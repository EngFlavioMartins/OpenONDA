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
match the present leapfrogging geometry. However, the paper explicitly excludes
flow instability from the subsequent parametric calculations (pages 2–3),
including the Fig. 5 trajectory. Its amplitude `epsilon/R0 = 0.05`, mode `n = 8`
perturbation belongs to the separate Fig. 3 validation example at `Re = 3415`
and is described as a displacement in the axial z direction. The older tutorial
uses a radial disturbance. Therefore Fig. 5 is an unperturbed kinematic
reference, not a matched validation of that seeded instability experiment.
The current LES qualification and baseline use zero imposed disturbance.

The paper uses periodic boundaries in a box of axial length `20 R0` and
transverse dimensions `7 R0` by `7 R0`, with `R0/dx=200`. Current treecode
induction is unbounded; the GBD computational lattice is not a periodic
induction boundary condition. This remaining difference prevents calling the
overlay a fully matched benchmark.

`assets/assess_lbm_agreement.py` compares field-core maxima at equal axial
positions, weights the two initially distinct cores equally, and refuses
extrapolation. It stops assigning identities when there are not exactly two
maxima or their straight-line bridge reaches the explicit diagnostic cutoff
(default 0.5 of the weaker maximum). That cutoff is not a physical merger
criterion. Sampler cadence, plane spacing and peak detection must still resolve the paths.

## Current study scope

The requested VPM comparison keeps LES, SSPRK3 and transposed stretching
enabled. The previous no-LES/RK4 transport controls were supporting diagnostics
and did not validate agreement with this reference. The automatic comparison
now reads SurfaceSampler VTS/PVD outputs; it does not reconstruct fields from
particle snapshots. See the [current workflow](../../readme.md).
