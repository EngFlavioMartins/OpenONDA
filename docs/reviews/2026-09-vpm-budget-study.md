# VPM study with a three-hour budget

## Actual outcome, 2026-09-08

No campaign processes remain running. The baseline and weak realignment
runs reached their wall-clock limits; neither reached a numerical health
stop or the intended late-time breakdown interval. The half-timestep run
ended without a recorded terminal event. Its stale `running` status has
been corrected to `interrupted`; the cause and exact final step are unknown.
The launcher therefore did not produce its automatic final report. The
comparison was generated afterward from the saved VPM sampler outputs.
The planned reporting deadline was missed.

| Run | Recorded end | Steps | Wall minutes |
|---|---:|---:|---:|
| LES baseline, GBD/Lagrange6 | t=2.8425 | 379 | 53.1 |
| Same configured baseline plus weak moment-preserving realignment | t=2.625 | 350 | 52.1 |
| Half timestep baseline, interrupted | last scalar t=.375; last field t=.3 | last scalar step 100 | not recorded |

On the common fixed interval x/R0=[.55, 2.5], equal-ring RMS radius error
is **5.039% of R0 for the baseline and 5.093% for weak realignment**.
The difference is below the observed sampling-cadence sensitivity; no
improvement from stabilization is demonstrated. All saved fields retain
two separated core maxima. Neither run covers both rings through x/R0=3.5.

**The three main runs have different recorded source fingerprints.**
Their trajectory errors are descriptive measurements, not a certified
controlled comparison isolating stabilization or timestep. The earlier
short timestep checks below used matching fingerprints; they do not
establish long-time convergence. GBD/Lagrange6 remains a provisional
viscous choice, not a demonstrated optimum.

The bounded experiment is inconclusive for the original objective of
recovering LBM dynamics through breakdown. No further simulations were
launched after this budget. A defensible continuation would first freeze
the solver source and establish that a paired baseline/stabilized run can
reach the required physical interval within the available runtime.

Generated evidence: `tutorials/vpm/vortex_interactions/figures/study/les/report.md`,
`lbm_agreement.json` and `core_trajectories.png` in the same directory.
Fields and self-diagnostics remain the solver's SurfaceSampler,
FlowIntegralsSampler and RingDiagnosticsSampler outputs. Analysis reads
those outputs; it does not reconstruct or rerun the flow from snapshots.

The sections below record the original plan and preliminary evidence.

Started 2026-09-08 at 09:36 UTC. Stop calculations by 12:16 UTC and finish
the report by 12:36 UTC. The user explicitly rejected a multi-day sweep.

All scientific comparisons retain LES (Cs=.20), SSPRK3, transposed stretching,
Re_Gamma=3000, zero imposed disturbance and the corrected Gaussian core.
Diagnostics and fields come from VPM samplers; no snapshot reconstruction.

## Allocation and decisions

- Runtime/timestep checks: at most 20 minutes.
- Baseline: at most 50 minutes.
- One stabilization comparison: at most 50 minutes.
- Focused numerical sensitivity check: at most 30 minutes.
- Analysis and final report: at least 20 minutes.

The remaining ten minutes provide startup/output margin. Run caps are checked
between accepted steps; final output may add some overhead.

The six-case viscous sweep was cancelled. A built-in timing probe attributes
98.9% of the first CS step to coupled evolution, with LES updating taking only
.0028 s. The tree descends whenever a node's core-width spread exceeds 1e-5
of its mean. LES core spreading creates unequal widths, making this setting
expensive. The initial suspicion about serial viscosity statistics was not
the dominant cost; its small parallel reduction change passed correctness tests.

Fixed-core GBD/Lagrange6 is the working baseline candidate, based on the prior
transport evidence and current runtime. This is not a claim that it is the
best viscous scheme. The h=.04, sigma=.04 probe takes about two seconds per
step initially; growing particle populations can increase this cost.

Check dt=.0075 against .00375 at equal t=.15 before choosing the long-run
timestep. Aim for t=6 or the first health stop. Compare the reference over
a common covered interval, and report an unavailable score if coverage is
insufficient. Select stabilization from the baseline's observed failure;
weak moment-preserving realignment is a candidate for divergence/misalignment,
not a remedy for excess physical or numerical diffusion.

## Required outcome

Deliver the baseline's observed trajectory, coherent-core interval and
termination; compare one justified intervention; report time/space sensitivity
where the budget permits. Distinguish budget stop, numerical health stop and
physical merger. A negative or inconclusive result is valid; longer survival
alone is not success. Full spatial convergence or matched-boundary validation
will not be claimed from this bounded study.

## Completed initial qualification

At t=.15, h=.04, sigma=.04, Cs=.20, the dt=.0075 and .00375
SurfaceSampler fields start bitwise identically. At the final time, their
relative L2 differences are .2338% for velocity and 1.0707% for vorticity.
Both sampled maxima have the same grid coordinates. Their leading peak
values are .8705 and .8863 of the prescribed initial peak, respectively.
This is whole-solver timestep sensitivity, including diffusion/remapping,
not an isolated RK tableau error or proof of long-time convergence.

The two runs took 68.7 and 109.3 seconds including initialization and output.
The main comparison uses dt=.0075; the half-dt check is budgeted separately.
All trajectory scores use fixed, predeclared intervals from x/R0=.55 to
1.5, 2.5, 3.5 and 5.5, with no extrapolation. Compare methods only on intervals
covered by both coherent-core histories.

The executable default is now `allrun.sh --campaign budget`: baseline and
weak moment-preserving realignment are capped at 50 minutes each, followed
by a 30-minute half-dt baseline check. Weak realignment remains a candidate
for divergence/misalignment, not an established optimized stabilization.
The tests for run caps, terminal sampler output, LES statistics and launcher
configuration pass.

The bounded launcher started at approximately 09:51 UTC. With all three
run budgets exhausted, calculations should end around 12:01 UTC plus final
output overhead, leaving time for analysis within the three-hour deadline.
The launcher automatically writes `figures/study/les/report.md`, the detailed
JSON score record and field/trajectory figures when the serial runs finish.
Lifecycle, sampler, backup, health and tutorial regression tests passed.

## Preliminary baseline result

The active baseline's SurfaceSampler fields through t=1.35 retain two
separated maxima and capture the first overtaking. Over the predeclared
x/R0 interval [.55, 1.5], the equal-ring RMS radius discrepancy from LBM is
1.8817% of R0 (per ring 1.9172% and 1.8456%; maximum discrepancy 5.1708%).
These are grid-resolved preliminary kinematics, not validation of the later
merger or an optimized stabilization. No axial phase fitting was applied.
Source run: `les_budget_leapfrog_gbd_l6_baseline_dt.0075_h.04`.
