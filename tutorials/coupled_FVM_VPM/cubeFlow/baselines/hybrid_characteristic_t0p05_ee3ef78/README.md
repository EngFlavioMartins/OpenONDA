# Rejected characteristic donor-boundary branch

This production-resolution run tested `donor_boundary_mode="characteristic"`
on cubeFlow's merged six-face coupling patch.  It was stopped after the first
coupling step at 0.05 s: the mixed inflow/outflow switch created a strong
outer-boundary vorticity sheet and handed off 397,999 particles, compared with
88,740 for the accepted Dirichlet baseline.

The generic characteristic mode remains available as an opt-in coupler API,
but it is unsuitable for this merged patch without face-specific treatment.
The lightweight logs are retained here; no scheduled force or field sample
had been reached when the run was stopped.
