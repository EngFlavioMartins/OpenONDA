# VPM leapfrogging: qualification, baseline, then stabilization

## Official tutorial continuation — 9 September 2026

The user requested an official baseline solution reproducible as the first
command in the tutorial's `allrun.sh`, followed by the stabilized runs.
The new comparison **started at 09:43 UTC on 9 September**, after the Metal
slot became free. The baseline finalized at its native time cap; stretching
viscosity is now running. The sequence uses `setup_les.py`
directly and writes native `solution/les_<variant>/` and
`samples/les_<variant>/` outputs. No historical data were imported as a
completed solution or overwritten. The earlier bounded result below is
retained as evidence motivating the new run.

Run order: baseline, stretching viscosity (.5), moment-preserving realignment
(frequency .384684814725/s), splitting (twice initial peak strength).
All have identical baseline physics and **1200 steps at dt=.0075, t=9 s**.
Native per-run caps are 150 minutes: ten hours for four serial caps, with
final-output/analysis overhead kept within approximately twelve hours of
execution. Queuing behind unrelated authorized work is separate from this
compute budget. A cap can prevent full physical coverage; it is not a blow-up.

The t=9 horizon implements the user's requirement that successful runs allow
both cores to reach x/R0=7. Saved baseline/viscosity tracks at t=3.3 have the
slow core near x/R0=3.98/3.92. Mean-speed extrapolation predicts arrival near
t=5.6, but least-squares speeds over t=2.4–3.3 give .600/.571 R0/s, predicting
t=8.33/8.69. Hence t=9 with a margin over the slower recent-motion estimate.
This is deliberately a rough estimate, not a fit to future dynamics. The
automatic report records each core's reached x and whether both reached7;
the fixed scoring intervals now include .55–7 without extrapolation.

Native backups every100steps and at finalization preserve an official
numerical solution even after a later interruption. Flow/ring diagnostics
remain every10steps; the primary half-plane remains every.15s at.02 spacing.
An orthogonal y=0 plane, covering both signs of z, is sampled natively every
.30s at.04 spacing and at termination. It supplies an additional deformation
check without tutorial-owned field reconstruction.

`allplot.sh` now targets these official paths and produces radius overlays,
fixed-interval LBM errors, scalar health/energy/enstrophy histories, and
common-time contours at1.5,3.3,4.5,6,7.5,9 when available. The detailed
[validation plan](../../tutorials/vpm/vortex_interactions/readme.md) uses
numerical survival and LBM fidelity together. Resource/time censoring and
inactive splitting cannot be counted as evidence of a winning technique.
The Fig.5 data do not independently certify instability breakdown.

The cleanup task's Fourier memory changes were confirmed in its work record:
axis-specific grid growth and componentwise padded operations, measured
diagnostic peak3.56→2.54GB with unchanged integrals and435tests passed.
The main installed OpenONDA package was refreshed from this checkout before
the new runs; its earlier installed copy lacked those changes. Running VLM
jobs use separate installed/frozen copies and were not modified. This task
did not edit numerical algorithms.

Validation:28 focused case/analysis/Fourier/style tests passed; the revised
t=9 configuration passed14 case/analysis/style tests. The plain installed
runtime also completed a one-step CPU smoke of the full-resolution baseline
in `/private/tmp/openonda-official-smoke-v5de0kz2`, with a native final HDF5
backup, completed metadata and both sampler planes. This smoke is not the
official production baseline. Existing battery scores were reproduced by
the updated analysis, and its new diagnostic figure was inspected.

- [x] Register the official baseline-first launcher and equal-physics stabilizers.
- [x] Set a time horizon allowing estimated travel to x/R0=7.
- [x] Verify installed runtime, native outputs and plotting paths.
- [x] Obtain a free Metal slot and launch the official baseline-first sequence.
- [x] Verify the first periodic native numerical backup in the tutorial.
- [ ] Complete all four official cases.
- [ ] Verify achieved coverage/survival and publish the final comparison.
- [ ] Send one consolidated thesis handoff after the final assessment.

### Authorized thesis handoff

After the tests and final assessment, send one message to the existing task
**Complete ch05 TODO study**, thread
`01a081a4-a01e-78b0-9dac-60770868f496` on the local host (thesis workspace:
`/Users/flaviomartins/Nextcloud/Thesis`). The user explicitly authorized this
handoff on9September. It has not yet been sent.

Ask that task to review and update Chapter3's VPM algorithm, equations,
physical model and stabilization descriptions against the implemented,
validated solver, then update the results discussion around Chapter5.1.3
(verify the actual section). The current thesis algorithm may differ
substantially and its stabilization coverage is incomplete. Supply the final
study record, relevant source/test pointers, official sampler results and
figures, separating demonstrated findings from unresolved limitations.

The thesis text has not been published: present the correct explanations,
equations and model directly, without an erratum, admission-of-correction
narrative or software debugging chronology. Let the thesis task translate
the evidence into academic, physics-first writing, consistent with its
notation. Preserve honest qualification of numerical survival, damping,
LBM agreement and unvalidated physical breakdown. Send this once, in a
single batch after results are assessed, and record delivery here before
pausing the follow-up.

The follow-up checks hourly and stays quiet during routine progress.
The previous Metal process80399 had ended before launch. A live process
inventory showed no production simulation and memory-pressure reporting
showed72% free. The cleanup task was notified that this battery now uses
the Metal slot; its unrelated jobs were not interrupted.

Actual launch: **2026-09-09 09:43:03 UTC**, detached serial shell
**PID5950**, caffeinate helper5951 and baseline child5953; console log
`/private/tmp/openonda-official-vpm-battery.log`.
The numerical runtime is `/opt/anaconda3/envs/OpenONDA/bin/python`, with
the installed memory fixes checked immediately before dispatch. It executes
the tutorial's existing `allrun.sh` from that directory, without PYTHONPATH
overrides. CPU-library limits are two threads, set externally; the cache is
`/private/tmp/openonda-official-metal-cache`. No official output directory
existed before launch. The four native caps nominally end around19:43UTC;
reserve final-output/analysis margin and finish within about21:43UTC.
Do not duplicate the sequence or reinstall its runtime while it runs.
Startup verification confirmed `arch=metal`, native status `running`, the
intended SSPRK3 configuration, initial flow/ring CSVs and both field-plane
VTS/PVD outputs in the official tutorial paths.

At the10:42UTC sparse check, the baseline is still running in the tutorial,
step588/t4.410 in the log. Native HDF5/XDMF backups100,200,300,400,500 are
present; the first HDF5 was opened successfully. Metadata records checkpoint
500/t3.75 with233778particles. Last scalar output is step580/t4.35,
N272475, CFL.05935 and relative divergence.02540. These are running-state
observations, not a final survival or LBM score. No further case is active.

At12:44UTC, the official baseline has finalized with native status
`wall_time_limit`: step968, t7.26, N386538, CFL.04588 and relative
divergence.02102. Its final numerical backup `vpm_000968.h5` opens normally;
both native plane indexes retain the terminal sample and existing VTS files.
This is the official tutorial baseline solution, with a budget stop rather
than a demonstrated blow-up. Its achieved x/R0 coverage and LBM agreement
will be assessed with the completed battery. The serial launcher proceeded
normally to stretching viscosity (child11351), now at step400/t3.0 with
N212160. Realignment and splitting remain queued; no duplicate was launched.

## Final bounded-study result — 9 September 2026

**Use GBD/Lagrange6, Smagorinsky LES Cs=.20, SSPRK3 and transposed
stretching as the practical baseline. Stretching viscosity with coefficient
.5 is the only tested addition that improved the measured LBM trajectory.**
This is a useful early-dynamics result, not validation through breakdown.
The fixed battery has ended; no further simulations are queued.

CS is working diffusively: in the short check its mean particle core grew
from .040000 to .053208 by t=.15, while enstrophy decreased 10.2%. The earlier
use of material-group radii could conceal this core growth because that
observable ignores Gaussian widths. No CS operator repair is supported by
this check. On the qualification interval x/R0=.55–1.5, CS's sampled-core
radius RMS was 2.577% of R0 versus GBD's 1.882%; GBD was also faster.

The subsequent matched battery gives:

| Method | Radius RMS, x/R0=.55–3.5 (% R0) | Observed effect |
|---|---:|---|
| GBD baseline | 5.715 | Reference control |
| Splitting | 5.725 | Zero splitting events; active splitting was not exercised |
| Stretching viscosity, coefficient .5 | **5.116** | Modest trajectory improvement, additional damping |
| Moment-preserving realignment | 5.725 | 450 events; no resolved trajectory improvement |

The stretching-viscosity reduction is .599 percentage points, about 10.5%
relative. Repeating the score with coarser saved-time sampling gives 5.827%
for baseline and 5.378% for stretching viscosity, preserving the ordering.
This checks output-cadence sensitivity only. Spatial/time-step convergence
and repeat-run variability were not established. Small differences between
the baseline and the inactive-splitting control should not be interpreted
as a splitting benefit.

At the last common scalar time t=3.375, baseline and stretching viscosity
have enstrophy 496.57 and 417.34, respectively, and relative divergence
.01457 and .01163. Thus the improvement accompanies appreciable extra
damping. The native sampler records stretching-viscosity coefficient .5;
its zero discrete-event count does not mean this continuous operator was
inactive. Realignment reduces the measured misalignment (.542 to .329
degrees) without improving the resolved radius trajectories.

All four saved scalar histories reach step 450, t=3.375. Splitting and
stretching viscosity finalized normally. Baseline was killed by SIGKILL
amid critical memory pressure, as recorded below. Realignment stopped
without finalizing metadata; its exit cause is not established. Baseline
and realignment retain native status `created` and last field time t=3.3;
their metadata must not be presented as completed runs. No native metadata
was rewritten to hide these outcomes.

The common field comparison ends at t=3.3 (t Gamma0/R0²=10.37). Recorded
planes show two deformed but separated vorticity maxima. Both core tracks
cover x/R0=.55–3.5, but not the predeclared interval through 5.5. **No run
demonstrated physical breakdown, a numerical blow-up prevented by
stabilization, or agreement with the LBM breakdown location.** The LBM
periodic boundaries also remain unmatched. This closes the half-day
experiment; the original full-breakdown validation remains unresolved.

The shortest supported continuation would be the baseline versus coefficient
.5 stretching viscosity only, after reducing or resolving the memory cost
of longer runs. It would need later common field coverage and one numerical
sensitivity check before claiming improved breakdown physics. No such
continuation is launched here.

Completed work:

- [x] Check CS diffusion and select the practical LES/RK3 baseline.
- [x] Run the baseline and three existing stabilizers; retain interrupted evidence.
- [x] Compare native sampler trajectories and fields on common coverage.
- [x] Record the supported choice, limits and reproduction commands.

Outputs are the [qualification report](../../tutorials/vpm/vortex_interactions/figures/study/les_qualification/report.md),
[battery report](../../tutorials/vpm/vortex_interactions/figures/study/les/report.md),
[trajectory comparison](../../tutorials/vpm/vortex_interactions/figures/study/les/core_trajectories.png),
and common-time field panels at t=3.3:
[baseline / realignment](../../tutorials/vpm/vortex_interactions/figures/core_sections/core_sections_leapfrog_t3.3_1.png)
and [splitting / stretching viscosity](../../tutorials/vpm/vortex_interactions/figures/core_sections/core_sections_leapfrog_t3.3_2.png).
All fields come from SurfaceSampler; scalar diagnostics come from the VPM's
flow/ring samplers. Analysis reads these outputs without particle-field
reconstruction. Case files and numerical source were left unchanged by
this final assessment; only plot labels and lifecycle wording were clarified.

For reproduction, use the existing `assets/study.py` command recorded below,
with `--steps 450` for the common bounded comparison and fresh output tags.
The original baseline requested 800 steps and was interrupted; that fact
must remain part of its provenance. Run methods serially. The clean
`setup_les.py` already exposes the same GBD/LES baseline physics, but its
older wall-time limits and variant list are not this battery's execution
record. Exact executed configuration is in each run's
`solution/vpm_metadata.json`.

The remaining sections preserve the campaign history and superseded plans;
the result above is the current decision.

## Resumed after the API cleanup — 8 September, 18:42 UTC

The user authorized continuing after the overhaul. Keep the case files
minimal: no restored metadata wrappers, exception handlers, gates or copied
runtime trees. Use the existing `assets/study.py` for research controls,
`solution/vpm_metadata.json` for solver-owned configuration/state, and the
native flow, ring and surface samplers. The cleanup task confirmed these
entry points and is leaving induction/diffusion/stabilization unchanged.

Current to-do list:

- [x] Short matched CS/GBD checks with LES, SSPRK3 and transposed stretching.
      Verify CS core growth and compare sampled field cores, not group radii.
- [x] Compare CS and GBD through the first leapfrog excursion (target t=1.5)
      and select the useful baseline, considering LBM error and runtime.
- [ ] Run that baseline plus three compatible existing stabilizers with
      identical physical/numerical inputs and the same requested horizon.
- [ ] Plot native sampler results, compare common covered intervals with LBM,
      and report whether any stabilization improves the baseline.

The practical budget remains about twelve hours of work, starting with this
resumption; reserve time for the final comparison. The earlier interrupted
snapshot campaign and its automation are superseded. No claim of a viscous
winner or an improved breakdown location is justified yet.

Checks use h=.04, sigma=.04,
dt=.0075, Cs=.20, unseeded Re_Gamma=3000 rings, theta=.5/order=3, a .0001
initial tail and native field output every .15. CS requests 20 steps with an
eight-minute cap. The next pair, `les_api_gbd_baseline_t15` and
`les_api_cs_baseline_t15`, requests 200 steps (t=1.5), with 25- and
70-minute caps respectively. The GBD history also supplies the short t=.15
comparison, avoiding a separate duplicate GBD run. Configuration,
lifecycle-related tutorial, core-spreading and sampler-analysis checks pass.
No case or numerical-source changes were required to start these checks.

The new CS sampler confirms active diffusion: at t=.075 the mean particle
core radius is .047131 (initial .040000), minimum .044480, maximum .048737;
enstrophy falls from 1968.863 to 1863.468 (5.35%). Effective viscosity remains
positive. Thus the current CS path is not inviscid. The group-ring major
radius uses particle positions and strengths, not the Gaussian core radii;
its continued oscillation alone is not evidence that diffusion is missing.
The LBM comparison uses the SurfaceSampler's vorticity-core maxima.

The post-overhaul CS check completed all 20 steps (t=.15), as recorded by
the solver's metadata. Mean particle core radius reached .053208 and
enstrophy fell to 1767.938 (10.2% below initial); relative divergence was
7.43e-5 and CFL .1973. No repair of the diffusion operator was indicated.

The serial GBD/CS first-excursion comparison was launched on 8 September
at approximately 18:50 UTC, supervisor PID 51803; console log:
`/private/tmp/openonda-vpm-viscosity-screen.log`. It runs the existing
`assets/study.py` twice with the common inputs above, GBD first, then CS.
The app follow-up `complete-twelve-hour-vpm-study` now follows this current
API workflow every 15 minutes, with a final deadline near 06:42 UTC on
9 September. Its obsolete snapshot instructions have been replaced.

At the 19:05 UTC check, GBD completed t=1.5 (200 steps; reported evolution
elapsed 11m56s). The equal-ring RMS radius error over x/R0=[.55,1.5] is
1.8817% of R0, reproducing the earlier first-excursion result. All recorded
planes retain two separated maxima. At t=.15, CS and GBD have identical
grid-resolved core coordinates, but normalized leading/trailing peaks are
.9392/.8858 for CS versus .8705/.8182 for GBD. This is a measured difference
in core damping, not yet a viscosity ranking. The serial CS run to t=1.5
has started; let it complete before selecting the baseline.

At 20:09 UTC, both sampled tracks already cover the entire predeclared
x/R0=[.55,1.5] interval: CS RMS radius error is 2.5773% of R0 versus
GBD's 1.8817%. CS remains numerically healthy and diffusive, but is slower
and does not improve this measured trajectory. Select GBD/Lagrange6 as the
practical baseline for the requested battery. Let the CS run finish at its
native limit before launching the battery; no GPU overlap.

The battery is `baseline`, `splitting`, `stretching_viscosity`, `p_moments`,
each independently from the same initial state with the common settings
above and GBD/Lagrange6 threshold .0001. Request 800 steps (t=6) and cap
each run at 130 minutes, leaving final assessment margin before 06:42 UTC.
Use tags `les_api_gbd_<method>_battery`. Keep the existing method settings
for splitting and stretching viscosity; use frequency .384684814725 for
weak moment-preserving realignment. Equal requested horizons do not imply
equal achieved horizons under cost/health stops; score common coverage.
This is a fixed short battery, with no additional parameter sweep.

The CS screen ended at its 70-minute cap, step 190 (t=1.425), with no
numerical health stop. The final sampler-based score remains 2.5773% on
the shared interval. The completed qualification report and trajectory
figure are in `figures/study/les_qualification/` under the tutorial.

The four-case battery started at approximately 20:26 UTC, supervisor PID
60071, log `/private/tmp/openonda-vpm-stabilization-battery.log`. Baseline
is advancing; the other three cases are queued serially. No case-file or
solver-source modifications were needed. The command for each method is:

```sh
python -u assets/study.py --scenario leapfrog --steps 800 --wall-minutes 130 \
  --dt .0075 --spacing .04 --integrator SSPRK3 --stretching TRANSPOSED \
  --core-ratio 1 --support circular --amplitude 0 --smagorinsky .20 \
  --initial-tail .0001 --tree-theta .5 --tree-order 3 --capacity 1000000 \
  --field-interval .15 --diffusion GBD --gbd-remeshing LAGRANGE6 \
  --diffusion-tail .0001 --frequency .384684814725 \
  --method METHOD --tag les_api_gbd_METHOD_battery
```

Here METHOD is baseline, splitting, stretching_viscosity or p_moments;
the runtime uses `/opt/anaconda3/envs/OpenONDA/bin/python` with Metal access.
Four caps total 8h40m, leaving approximately 1h35m for final output and
assessment before the overall deadline. No additional cases are queued.

### Resource interruption and shortened remaining battery

At 22:12 UTC the shell recorded `Killed: 9` for baseline PID 60073.
Its native metadata remains `created` because SIGKILL prevents finalization;
that stale status is not a statement that no simulation occurred. The last
saved scalar state is step 450, t=3.375, N=221335, divergence .01457,
misalignment .5424 degrees and CFL .0910. The log subsequently records
N=223617 but no accepted final time; do not invent its terminal step.

macOS memory-pressure events progress from warning at 22:11:32 UTC to
critical at 22:11:55, then normal at 22:12:08. The process exit is recorded
at 22:11:59. This strongly suggests a resource-related kill, not a reported
VPM numerical health failure; no log directly identifies the killing actor.
The machine has 16 GiB RAM. Original outputs and metadata are preserved.

Only the unstarted splitting, stretching_viscosity and p_moments runs are
continued, now requesting **450 steps (t=3.375)** to match the saved baseline
coverage and avoid deliberately entering its failed late-time interval.
All evolution, resolution, sampling and stabilization settings remain as
listed above; the native 130-minute cap is retained. The change in requested
horizon is a resource-driven deviation, so the final result must be limited
to common saved coverage and cannot establish agreement through breakdown.
The original serial shell stopped; the replacement serial log is
`/private/tmp/openonda-vpm-stabilization-remaining.log`.

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
