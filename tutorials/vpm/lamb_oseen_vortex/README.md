# Lamb–Oseen vortex

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
[RWM_STATISTICAL_METHODOLOGY.md](RWM_STATISTICAL_METHODOLOGY.md).
