# Lamb–Oseen vortex

## Console output

The campaign labels the start and completion of each physical case, method,
RWM aggregation, and validation phase. A failure prints the active phase and
preserves the command's exit status; later cases do not run.

Each solver prints its method, backend, integrator, timestep, particle capacity,
output schedule and log path once. Accepted-step progress shows the step count,
flow time, elapsed time and current particle count at roughly 30-second
intervals, checked at step boundaries. Scheduled diagnostics also count as
progress updates. The final accepted step and run outcome remain visible.

Energy, strength, impulse, enstrophy, helicity and centroids are printed only
when the existing flow-integral sampler is due. Warnings are immediate. The
full configuration remains in each case's `vpm.log`, while CSV and field-output
schedules are unchanged. No extra physics evaluations are performed for the
progress display.

## Diffusion workspace

The finite vortex columns diffuse along their axes as well as radially.
`setup.py` reserves a cumulative heat-kernel margin of
`3.6 * sqrt(4 * viscosity * TOTAL_TIME)` beyond the initial support when
sizing the solver domain. DVH's per-transfer grid padding is additional
workspace, not a substitute for that physical spread. The field-sampling
window is configured separately.

On GPU the diffusion grid is allocated once. DVH checks that every source's
heat support fits, including sources regenerated into the padding halo;
an insufficient allocation raises an error rather than truncating diffusion.
Changing the duration, spacing or domain therefore requires checking repeated
diffusion events, not only the first event.

## Post-processing the Random Walk Method

The Random Walk Method (RWM) represents viscous diffusion by giving every
particle an independent Brownian displacement,
\(\Delta\boldsymbol{x}=\sqrt{2\nu\Delta t}\,\boldsymbol{\eta}\). Consequently,
one RWM calculation is one Monte Carlo realization of the solution, not the
mean flow itself; its particle-to-particle scatter is numerical sampling noise,
not physical turbulence.

We therefore run the same case with several independent random seeds. At every
stored physical time, each realization is projected onto the same
cross-sectional grid and the signed vorticity components are averaged. The
velocity and flow diagnostics—such as vortex position, core radius, separation,
and orientation—are then obtained from this ensemble-mean field. Nonlinear
features are deliberately extracted *after* field averaging, and no averaging
in time is used, because temporal smoothing would blur the evolving vortices.

## RWM markers and uncertainty margins

Each plotted marker is the estimate obtained from the ensemble at that output
time. The shaded margin around it is a two-sided 95% confidence interval for
that estimate: Student's \(t\) interval is used for field and linear integral
quantities, while a delete-one-realization jackknife is used for nonlinear flow
features. The latter repeats the complete field-averaging and feature-extraction
procedure, so the interval includes the sensitivity of the diagnostic itself.

The margin is **not** the spread of particles, a physical fluctuation, or an
error bar relative to the analytical or reference solution. It measures only
the uncertainty caused by using a finite number of random realizations. A wide
margin means that the diagnostic is sensitive to RWM sampling and would require
more independent seeds for greater precision; when the margin is hidden by a
marker, the interval is simply smaller than that marker at the plotted scale.

The complete definitions and implementation choices are documented in
[rwm_statistical_methodology.md](references/rwm_statistical_methodology.md).
