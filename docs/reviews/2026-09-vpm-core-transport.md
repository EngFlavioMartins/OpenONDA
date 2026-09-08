# VPM leapfrogging: qualification, baseline, then stabilization

## Status and agreed pathway — 8 September 2026

**Latest clarification:** further simulations are authorized within about
12 hours total. The active plan is the [half-day continuation](2026-09-vpm-halfday-study.md):
an isolated-source baseline through later dynamics, then an intervention
chosen from its observed defect and a focused numerical sensitivity check.
The earlier short-budget result below is historical and inconclusive.

**Runtime constraint:** the user subsequently required results within 2–3
hours. The default is now a bounded GBD/Lagrange6 LES comparison with native
50/50/30-minute run caps, rather than the six-case qualification sweep.
See the [budget study record](2026-09-vpm-budget-study.md).

**The no-LES/RK4 detour did not establish an improved LBM match.** It exposed
useful numerical errors, especially coarse-timestep damping, but extending it
into a separate long campaign was not justified by the agreed objective.
Its continuation and queued helper runs were stopped. Abandoned campaign data
and helpers were deleted at the user's request. The deletion inventory is in
[2026-09-vpm-cleanup.json](2026-09-vpm-cleanup.json); compact historical evidence
is in [2026-09-vpm-supporting-controls.json](2026-09-vpm-supporting-controls.json).
These retained numbers are supporting observations, not independently
rerunnable raw datasets or LES validation.

The active pathway is:

1. Qualify the viscous scheme with **LES, SSPRK3 and transposed stretching**.
   Compare CS, GBD/M4' and GBD/Lagrange6 with identical initial particles,
   viscosity, LES coefficient, spacing, tree settings and physical times.
   Check timestep sensitivity before interpreting core damping or merger.
2. Run the selected unstabilized baseline far enough to compare with the LBM
   trajectory and identify its actual failure or loss of physical accuracy.
3. Test stabilization motivated by that failure with the baseline settings
   held fixed. Judge common-time fields, trajectory error and credible
   extension, not step count alone. A baseline need not blow up; numerical
   health failure and physical deformation/merger are different outcomes.

**No viscous candidate is currently certified as best, and no stabilized
configuration has demonstrated agreement through LBM merger/breakdown.**

## Useful findings retained from the supporting controls

- The initial Gaussian tail must be retained: discarding 5% and renormalizing
  raised a prescribed peak of 100 to about 107.27. Retaining all but .01%
  recovered about 99.90 in the checked initial profile.
- At t=.15, h=.03 and sigma=.04, no-LES GBD/Lagrange6 leading-core peaks were
  76.41 at SSPRK3 dt=.015, 93.28 at .0075, 96.49 at .00375 and 96.73 at .0015.
  Thus coarse RK3 damping invalidated the earlier apparent-merger claims.
  This supports refining RK3, not replacing the requested baseline with RK4.
- The production GBD diffusion-only control agreed with Gaussian heat
  evolution to about .2% over the checked interval. Six-point Lagrange
  remapping reduced translating-core damping compared with M4', but those
  DNS controls do not select the best LES scheme.
- GBD supports variable effective viscosity nu+nu_t. CS and GBD are therefore
  candidates for the requested LES comparison. GBD's repeated remapping and
  CS's growing spherical cores introduce different transport errors.
- The near-core tree sampling fixes and Gaussian/remapping unit tests remain
  useful implementation work. They were retained with the original LES
  stabilization audit and its results.

Historical settings theta=.5/order=3 were measured to have small tree errors
on selected DNS snapshots. They are a cost choice, not an error bound for the
LES baseline. Tree, spacing and pruning sensitivity must be checked on the
relevant LES states before declaring physical agreement.

## Reproducible workflow

The [tutorial launcher](../../tutorials/vpm/vortex_interactions/allrun.sh)
retains an optional short LES qualification campaign: Cs=.20, h=.03, sigma=.04, SSPRK3 dt=.00375
and dt/2, to equal t=.15. The bounded default instead uses h=.04, sigma=.04
and dt=.0075 with GBD/Lagrange6 as a provisional candidate; it does not certify
a viscous winner. Baseline and
stabilized campaigns require an explicit viscous selection; see the
[tutorial README](../../tutorials/vpm/vortex_interactions/readme.md).

All new runs configure the VPM FlowIntegralsSampler, RingDiagnosticsSampler
and SurfaceSampler, together with the solver's normal self-diagnostics.
The tutorial no longer writes auxiliary particle NPZ snapshots. Plots and
reference comparisons read CSV/VTS/PVD sampler outputs without reconstructing
fields or continuing a simulation from an auxiliary snapshot.

The meridional plane records velocity and curl-derived vorticity, initially,
every .15 physical seconds and at termination. Core maxima are resolved on
that saved grid. Material-label ring diagnostics are retained as proxies,
not substituted for field-core trajectories after groups mix.

## Reference and acceptance limits

[Cheng, Lou and Lim (2015)](https://doi.org/10.1063/1.4915890) Fig. 5 uses an
unperturbed Re=3000 case. Its loss of repeated leapfrogging includes viscous
core deformation and merger. The seeded mode-eight example is a different
case; the old tutorial's radial seed must not be compared as if it were the
same benchmark. LES remains enabled in the requested VPM study.

The LBM domain is periodic while the present VPM induction is unbounded.
This remains a physical setup discrepancy to assess, not hide by fitting an
axial shift, viscosity or time scale. Only the documented initial midpoint
origin adjustment is applied to the digitized trajectory.

Acceptance requires common-distance core-radius agreement through the
coherent phase, comparable deformation/merger location, and robustness to
numerical and sampler refinement. A peak-count or bridge cutoff does not
certify three-dimensional breakdown. Health stops, particle caps, rejected
stabilization events and actual physical merger must be reported separately.
