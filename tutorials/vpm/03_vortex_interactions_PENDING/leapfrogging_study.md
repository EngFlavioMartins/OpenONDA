# Leapfrogging-ring accuracy and stability study

Started 2026-09-13. Status: original baseline and targeted CS investigation
completed; baseline acceptance gate not met. The overall study is incomplete.
Under the requested ordering, stabilization testing waits until the
unstabilized physics passes the baseline gate. This file is the single study report and
authoritative experiment ledger.

## Reproducibility

- Starting branch: `development`.
- Starting HEAD: `98479c375eb773f0683df5a41cac5bea301671ec`.
- The starting checkout contains pre-existing uncommitted changes. They are
  retained. The starting diff, status, Python/package environment, full original
  configuration, source hashes, and source archive are in
  `solution/leapfrog_accuracy_20260913/provenance/`.
- Python: `/opt/anaconda3/envs/OpenONDA/bin/python`, 3.11.15, macOS arm64;
  Taichi 1.7.4, NumPy 2.4.6, SciPy 1.17.1.
- Tutorial-directory invocations without `PYTHONPATH` import the installed
  `site-packages` solver. Study commands explicitly select the development
  checkout; historical installed-generation results are not fresh controls.

Run from this tutorial directory:

```sh
PYTHONPATH=/Users/flaviomartins/OpenONDA python setup_les.py \
  --variant baseline --case-name study_original_20260913
```

The output name is the only case override. Physics: R=1, circulation=pi per
ring, Re=3000, nu=pi/3000, physical Gaussian core=.1, axial separation=1,
zero imposed disturbance. Numerics: h=.06, sigma=.06, compensated Gaussian
initialization, retained tail=1e-4, dt=.0075, SSPRK3, treecode theta=.5/order=3,
transposed stretching, CS, Smagorinsky Cs=.20, no stabilization, AUTO backend,
f32, 1200 requested steps (t=9), 24-minute native wall cap. Native health and
resource guards remain as supplied; their stops will not be called blow-up.

## Reference limits

The supplied `assets/references/leapfrogging_lbm_trajectory.csv` contains only
ring ID, x/R0, and R/R0. It supplies no time coordinate or breakdown fields.
Its provenance identifies the unperturbed Re=3000 Fig. 5 problem in
[Cheng, Lou & Lim (2015)](https://doi.org/10.1063/1.4915890); the seeded
Re=3415 instability example is a distinct physical case. LBM time/phase errors
and matched three-dimensional breakdown cannot be inferred from this CSV.
The only coordinate adjustment is the documented axial-origin shift of 2.5 R0.
Reference data are not modified.

## A. Original baseline results

A0 retains two resolved meridional core maxima through its last sample at
t=6.0825. Their final positions `(x/R0,R/R0)` are `(7.060,1.180)` and
`(7.740,.740)`, with a bridge-to-weaker-peak ratio .104. The native stop is
**resource_limit**, because available memory reached 1.97 GiB; it is neither
physical breakdown nor numerical blow-up. Numerical survival is only bounded
below by 6.0825. The requested t=9 horizon was not completed in this run.

| Common axial interval x/R0 | Radius RMS / R0 | Every-other-output RMS / R0 |
| --- | ---: | ---: |
| .55–1.5 | .02305 | .01812 |
| .55–2.5 | .06071 | .05288 |
| .55–3.5 | .08615 | .08172 |
| .55–5.5 | .16600 | .16281 |
| .55–7.0 | .18845 | .18356 |

Errors are equal-ring, uniform-distance scores, without phase fitting or
extrapolation. Coarser output sampling changes the scores by up to .00783 R0;
it does not remove the growing discrepancy. Peak coordinates are resolved on
the .02 R0 sampler grid. Late reference paths approach each other and do not
provide time-resolved identities through three-dimensional breakdown.

Passages occur at t=1.16250 [1.05,1.20], 3.43125 [3.30,3.45], and
5.653125 [5.55,5.70]. Successive passage intervals are 2.26875 and 2.221875 s;
the observed two-passage cycle is 4.490625 s (frequency about .22269 Hz).
Radial separation at those passages stays near .64–.655 R0. LBM passage
times, period/frequency, and temporal phase error are unavailable from the CSV.

The discrepancy is a growing **downstream shift of radial oscillations**,
followed by persistent coherent oscillation where the supplied reference paths
converge. The first leading-ring radius maximum is 1.280 R0 versus 1.276 R0
in LBM, so the initial major-radius excursion is close while its axial phase
already lags.

| Corresponding radius maximum | VPM x/R0 | LBM x/R0 | Downstream shift/R0 |
| --- | ---: | ---: | ---: |
| Initially leading core, first | 1.540 | 1.368 | +.172 |
| Initially trailing core, first | 4.260 | 3.587 | +.673 |
| Initially leading core, second | 6.800–6.860 | 4.915 | +1.885–1.945 |

The leading core's peak-to-peak axial wavelength is 5.260–5.320 R0 versus
3.5465 R0 in LBM: 48.3–50.0% longer. The range records a sampled flat peak,
not a statistical confidence interval. The following first minimum is also
displaced (x≈4.30 versus 3.753). These spatial offsets are not converted into
invented time offsets; digitized/field maxima are recorded in
`solution/leapfrog_accuracy_20260913/spatial_extrema.json`.

At termination N=16,104 and all stabilizer event counts remain zero. Mean
particle sigma has grown from .06 to .22389 (range .19878–.24697); recorded
energy and enstrophy are .71873 and .19116 of their initial values. Axial
impulse changes by +.0147%. Final normalized divergence=.01404,
misalignment=.485 degrees, and Lagrangian CFL=.05975, all below their native
limits (.12, 25 degrees, 1). Material-ring circulation proxies are 3.103 and
3.132; these are not independent circulation measurements of merged cores.
The existing native mode estimator gives final axial mode-eight amplitudes
below .00097 R0 and radial amplitudes below .00251 R0. The orthogonal-plane
mirror-asymmetry metric is .01952. These diagnostics and the two surviving
core maxima do not establish physical breakdown; mirror symmetry also cannot
rule out all non-axisymmetric modes.

See [trajectories](figures/leapfrogging_study/core_trajectories.png),
[passage history](figures/leapfrogging_study/leapfrogging_history.png),
[native diagnostics](figures/leapfrogging_study/diagnostic_histories.png), and
[sampled core sections](figures/leapfrogging_study/core_sections/).

### Equivalent continuation and endpoint

Reducing only inactive particle capacity to 120k reproduces A0 bitwise through
t=6. Strict native restarts extend this same trajectory past the original
memory stop. The first continuation reaches its 24-minute segment cap at
t=8.49. Resuming its saved state reaches **resolution_lost at step 1178,
t=8.835**: normalized divergence=.120326 exceeds the supplied .12 threshold.
Misalignment=6.212 degrees and CFL=.05368 remain below their limits. No NaNs
or unbounded energy were observed; this is a numerical health endpoint, not
an observed catastrophic blow-up time. Resource-censored segments do not
count as stability failures.

A fourth passage occurs at t=7.75313 [7.65,7.80], giving an additional passage
interval of 2.10000 s and a two-passage cycle of 4.32188 s. At the final sample,
the two core maxima remain identifiable at `(10.080,.860)` and `(10.720,1.080)`;
their normalized peaks are .14562 and .14871, with bridge ratio .34548.
The [final section](figures/leapfrogging_study/core_sections/core_sections_leapfrog_Re3000_t8.835_1.png)
shows broad rounded cores, not a demonstrated loss of coherent rings.
Mode-eight axial amplitudes are .00513/.00383 R0 and radial amplitudes
.00458/.00307 R0; the mirror-asymmetry metric is .04822. These do not establish
matched physical breakdown. Other azimuthal modes are not resolved by this
single-mode summary.

Final mean sigma=.25653, energy/initial=.67391, enstrophy/initial=.14603,
axial impulse change=-.2491%, maximum particle strength/initial=3.0837,
and material circulation proxies=3.281/3.170. Particle count remains 16,104;
all stabilizer counters remain zero. No baseline is frozen on this evidence.

### Native energy-estimator limitation

The CSV identifies energy as `periodic_fourier_energy`, evaluated on an
internally managed Fourier grid; it is not a direct measurement of the
unbounded-domain kinetic energy. Grid state/history is initialized afresh on
restart. At t=3.75, A0 and A1 have bitwise-identical particle fields and equal
enstrophy/impulse, yet logged energy differs by .8408%. The CSV marks grid
transitions via `kinetic_energy_rate_source`; the evolution code manages
`_fourier_grid` separately from the restarted particle state. Thus the energy
ratios above are **native estimates**, and restart offsets or grid-transition
bumps are not evidence of physical damping/growth. The plots label this
limitation. No energy-triggered stabilization is active, so it does not alter
these trajectories. `restart_diagnostic_difference.json` records the check.

## Experiment ledger

| Run | Phase | Question/Hypothesis | Single change | Value | Result/metrics | Interpretation | Next decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0-launch | A | Can the supplied AUTO backend initialize inside the sandbox? | None | AUTO/Metal | Failed before particles or time integration: Metal host-to-device map assertion | Environment initialization failure, not numerical blow-up | Retry identical numerical case with host GPU access |
| A0: study_original_20260913 | A | Does the current development source reproduce the supplied CS baseline discrepancy? | None; new output namespace only | Supplied setup_les baseline | 811 steps, t=6.0825, resource_limit: available memory 1.97 GiB <= 2 GiB; 19m11s native elapsed; no numerical failure observed | Original dynamics reproduced; survival is right-censored by host resource availability, not blow-up | Preserve original outputs; use shorter causal contrasts; no stabilization ranking |
| B-op | B, instantaneous diagnostic | Do grown radii themselves alter self-induced speed? | Radius arrays only at fixed A0 positions and strengths; no time evolution | Actual, molecular-only sqrt(.06²+4 nu t), and initial .06 at t=0,.75,1.5,3 | At t=1.5 actual self axial speeds .610822/.832190; molecular-only .673783/.875305 m/s; mutual .885640/-.027220 changes negligibly | LES core growth alone changes near-core induction; resetting sigma is a diagnostic, not a proposed physical model | Verify the growth law and isolate time-integration/LES effects before accepting a baseline |
| B-law | B, operator qualification | Is the Gaussian core update or LES scaling implemented incorrectly? | None; evaluate existing kernels against their analytical expressions | f32, 400 half steps at dt/2=.00375; fixed viscosities; analytical Smagorinsky formula | Maximum relative sigma² error 4.52e-6; LES formula error 2.36e-7 | Consistent with f32 rounding; no growth coefficient or viscosity scaling defect demonstrated | Do not change the CS formula; isolate LES in a trajectory contrast |
| B1: study_cs0_20260913 | B | Is the LES contribution responsible for the regular-stage phase/trajectory error? | Smagorinsky coefficient only | .20 → 0; same initial particles, CS and transposed SSPRK3; t=3 diagnostic horizon | resource_limit at step 1, t=.0075: available memory 1.94 GiB <= 2 GiB; no usable trajectory contrast | Environment-censored attempt; not evidence about LES or numerical failure | Qualify a smaller inactive capacity before retrying |
| A1: study_capacity120k_20260913 | A/B resource qualification | Can a smaller inactive allocation reproduce A0 and permit causal controls? | Allocated maximum particles only | 1,000,000 → 120,000, same 16,104 initial particles; t=3 | Completed 400 steps; every particle field in four checkpoints and velocity/vorticity in all 21 core planes are bitwise equal to A0 | Inactive capacity is a storage-only change for this fixed-population control | Retain 120k allocation for causal controls; retain 2 GiB memory guard |
| A1-continuation | A, survival observation | Was the original stop solely resource censorship, and where does the equivalent control actually lose usability? | Observation horizon only; strict native restart from A1 step 400 | t=3 → requested t=9; 800 additional steps | wall_time_limit at step 1132, t=8.49; 24m native wall cap | Native numerical health versus physical morphology versus resource/cap stops | Resume the saved state for the remaining 68 steps to t=9 |
| A1-tail | A, survival observation | Does the equivalent control encounter a health limit before t=9? | Observation horizon only; strict restart after wall cap | t=8.49 → requested t=9; 68 additional steps | resolution_lost at step 1178, t=8.835; divergence .120326 > .12; finite fields, misalignment 6.212 degrees, CFL .05368 | Health limits and all physics remain unchanged | Record numerical health loss separately from unvalidated physical breakdown |
| B-quadrature | B, initial-state diagnostic | Is the initial particle spacing already producing a large core-field error? | Particle spacing only, physical core and numerical sigma fixed | h=.06 → .04, sigma=.06; 162 common direct-field probes | Velocity relative L2 1.374e-4; curl relative L2 5.510e-4; peak 100.056 → 99.934 versus nominal 100; N=16104 → 49784 | Initial represented core/induction are close; no time-evolved spatial convergence is claimed | Lower priority of an initial-field mismatch; test evolution/LES |
| B-tree | B, saved-state diagnostic | Could tree approximation explain the measured core-field discrepancy? | Replace saved tree field evaluation by direct Gaussian sums, identical particle state/probe points | A0 step 400, t=3; 100 high-vorticity plane probes | Velocity relative L2 3.046e-5; curl relative L2 7.472e-6 | Small source-field error on this state; full transport/stretching convergence is not established | No tree-parameter trajectory run justified by these probes |
| B1-retry: study_cs0_cap120k_20260913 | B | Is the LES contribution responsible for the regular-stage error? | Smagorinsky coefficient only versus A1 | .20 → 0; qualified 120k capacity; t=3 | Completed t=3 (4m50s); radius RMS .55–2.5: .06850 versus .06071 control; first field-core passage 1.15588 versus 1.16250; final sigma .12715 | Compare common-distance radius error, first passage, field shape and sigma growth; Cs=0 is diagnostic only | Removing LES worsens early spatial agreement; keep intended Cs=.20, do not fit it to the reference |
| B1-post | B, saved-state diagnostic | Do spherical kernels still dominate core width without LES? | Reuse the B-op diagnostic on saved Cs=0 states; no new trajectory | t=1.5 and 3 | At t=3 the isotropic-kernel fraction of meridional width is 62–70%, versus 73–80% with LES | Molecular CS retains substantial finite-kernel deformation error | Do not attribute the discrepancy exclusively to LES |
| B2: study_halfdt_cap120k_20260913 | B | Does frozen-over-step LES / CS splitting dominate regular-stage error? | Timestep only versus A1 | .0075 → .00375; t=3 | Completed 800 steps, t=3, native elapsed 13m41s; local vorticity difference 12.01% at t=1.5 | Compare identical-time sampled fields and first passage, plus LBM radius scores | Local fields are not converged at nominal dt; use B3 to establish a shrinking difference, then extend the existing halfdt control through the reference window |
| B3: study_quarterdt_cap120k_20260913 | B | Does the 12% local vorticity difference at t=1.5 decrease with another timestep refinement? | Timestep only versus B2 | .00375 → .001875; stop at t=1.5 | Completed 800 steps, t=1.5, 11m41s; final adjacent-step field differences shrink about 7×: velocity 2.672% → .381%, curl 12.011% → 1.730% | Identical-time fields and material-centroid passage, using three timestep levels | Establish a shrinking field difference or retain unresolved timestep sensitivity; no further timestep sweep |
| B2-continuation | B | Does the large later spatial-phase discrepancy persist at the finer timestep? | Observation horizon only; strict restart of B2 | t=3 → requested t=6.3, enough for the common x≤7 reference window; 880 additional halfdt steps | wall_time_limit at step 1660, t=6.225, with both cores beyond x=7; full-window RMS .18955 versus .18845 original; wavelength remains 48.9–51.1% too long | Full-window radius scores and corresponding spatial maxima, compared with A0/A1 | Large spatial phase discrepancy persists despite timestep refinement; reference-window question answered, no further horizon extension needed |
| B-RK | B, tableau analysis | Can rapid core rotation make the nominal SSPRK3 timestep sensitive locally? | Evaluate the current tableau at the three tested dt values; no trajectory | Initial straight-core estimate Omega=50/s | Exact SSPRK3 stability polynomial; frozen-rotation amplitudes at t=1.5 are .85454, .97985, .99744 | Compare numerical damping with exact rigid rotation, whose amplitude is constant | Interpret local field sensitivity without changing the integrator or predicting actual ring shrinkage from a frozen model |


## Initial ranking of diagnostic hypotheses

1. Finite and growing particle cores change induction and deformation at h=.06;
   insufficient initial overlap (sigma/h=1) and fixed spherical basis transport
   may bias the physical core evolution.
2. Active LES contributes nonuniform diffusion; isolate its role only after
   measuring the effective viscosity and physical core growth in A0.
3. An initial-field mismatch, time integration/splitting error, or induction
   approximation could imitate a CS defect; use short matched controls if
   evidence warrants them.
4. Boundary/model mismatch: unbounded VPM versus periodic LBM. This cannot be
   removed by tuning CS, and reference time/breakdown evidence is absent.

The implemented Gaussian core update is sigma_new^2 = sigma_old^2 +
4 nu_eff dt. The accepted update brackets SSPRK3 with two half CS steps.
Smagorinsky uses particle-volume^(1/3), not the evolving sigma, as filter width.
The kernel checks verify these expressions. Integrating the actual A0 mean
effective-viscosity samples gives growth ratios `(mean sigma²-sigma0²) /
(4 integral mean nu_eff dt)` of 1.00120, 1.00069, 1.00061 and 1.00041 at
t=.75, 1.5, 3 and 6. The integral uses the native ten-step output cadence,
so this is approximate quadrature evidence, not a per-step exact replay.
There is no evidence of a factor-of-two/four error or duplicate CS application.

### What a correct CS law does and does not establish

The kernel qualification verifies the heat equation for the chosen Gaussian
convention, not consistency of a large, spherical kernel in a deforming flow.
For a Gaussian in constant incompressible linear strain with directional rate
`s`, write its directional squared width as `a²`. The exact advection-diffusion
width is `a_exact² = a0² exp(2st) + (2 nu/s)(exp(2st)-1)` (the limit at s=0
is `a0²+4 nu t`). Advecting particle centres while keeping their kernels
spherical gives `a_CS² = (a0²-sigma0²)exp(2st) + sigma0² + 4 nu t`.
Thus the initial strain response omits `2s sigma0²`, even if quadrature is
perfect. This is a finite-kernel modelling/discretization limitation, not a
wrong factor four in diffusion; as sigma0 tends to zero, the undeformed
accumulated diffusion width still leaves a finite-time strain error.

For this initialization sigma0²/a0²=.36. A positive-weight meridional
second-moment proxy from A0 attributes 62–74% of the represented core width
to isotropic particle kernels at t=1.5, and 73–80% at t=3. This proxy uses
material groups and is restricted to the coherent, nearly axisymmetric stage;
it is not an independent physical-breakdown criterion. Together with the
fixed-state induction contrast, it supports a CS finite-core explanation.
It does not by itself assign all observed LBM error to that mechanism.

The current code freezes LES viscosity over a macro-step, despite symmetric
placement of its CS half steps. Consequently, third-order SSPRK3 transport
does not establish third-order accuracy of the full CS/LES update. The B2–B3
contrasts below measure this full-scheme timestep sensitivity.

### LES contrast: smaller cores do not fix the trajectory

B1 changes only Cs=.20 to zero against the allocation-qualified control,
with identical initial particles. Both complete the common t=3 observation.

| Quantity | Intended LES control | Cs=0 diagnostic |
| --- | ---: | ---: |
| Radius RMS/R0, x=.55–1.5 | .02305 | .02747 |
| Radius RMS/R0, x=.55–2.5 | .06071 | .06850 |
| Same .55–2.5 score using every other field output | .05288 | .06504 |
| First sampled-field passage, s | 1.16250 | 1.15588 |
| First material-centroid passage, s | 1.162763 | 1.157350 |
| Mean particle sigma at t=1.5 | .13946 | .09941 |
| Mean particle sigma at t=3 | 0.17533 | .12715 |

Field passages share the [1.05,1.20] output bracket. The material-centroid
check reuses the same crossing calculation on existing ring-diagnostic CSVs
with a .075 s cadence; it is a separate material proxy, not a subgrid field
peak measurement. Cs=0 does not provide both cores through x=3.5 by t=3, so no
longer-distance RMS or full-cycle frequency comparison is manufactured.

The first passage changes little while the downstream radius error becomes
larger. Thus extra LES diffusion is not the leading explanation of the early
spatial discrepancy, and removing it is not a supported correction. At t=3,
even the molecular-only calculation attributes 62–70% of its represented
meridional width to spherical particle kernels, versus 73–80% in the intended
LES control (`core_induction_cs0.json` and `core_induction.json`). The
finite-core strain/induction limitation remains when LES is absent. These
short controls do not establish late-time or breakdown behaviour for Cs=0.

### Timestep sensitivity

B2 halves dt from .0075 to .00375 with Cs=.20 and the same initial particles.
The initial sampled fields are bitwise identical. At t=.75, 1.5 and 3,
velocity relative-L2 differences are .5509%, 2.6720% and 2.4951%; vorticity
differences are 2.5480%, 12.0113% and 10.6137%. These compare the same native
sampler points at equal physical times, normalized by the finer field norm.
The nominal timestep is therefore not demonstrated to converge local fields.

Yet the .55–2.5 radius RMS barely changes, .06071 → .06056 R0. The first
material-centroid passage, resampled to the same .075 s cadence, shifts only
1.162763 → 1.162235 s. The grid-peak passage shifts 1.16250 → 1.15714 s within
its common [1.05,1.20] bracket. B3 refines once more to .001875 and completes
t=1.5, without changing the initial sampled fields.

| Time, s | Velocity difference, nominal→half | Velocity difference, half→quarter | Vorticity difference, nominal→half | Vorticity difference, half→quarter |
| --- | ---: | ---: | ---: | ---: |
| .75 | .5509% | .07635% | 2.5480% | .35649% |
| 1.5 | 2.6720% | .38082% | 12.0113% | 1.72972% |

The approximately sevenfold reduction is strong evidence of a shrinking
timestep error over these levels, not a proof of global asymptotic order for
the full nonlinear split scheme. Material-centroid passage times converge
1.162763 → 1.162235 → 1.162166 s on the common .075 s diagnostic cadence.
The finest early radius RMS is .02494 over x=.55–1.5, versus .02624 at halfdt
and .02305 nominal; it does not remove the early spatial discrepancy.

The halfdt control is continued to t=6.3 to cover the same x≤7 reference
window as A0. This is an observation extension, not another timestep value.
The completed comparison below shows that the later wavelength discrepancy
persists; this conclusion is not extrapolated from the short controls.

A relevant numerical mechanism is resolved analytically from the current
SSPRK3 tableau: for rigid rotation, `R(z)=1+z+z²/2+z³/6` and
`abs(R(i theta))²=1-theta⁴/12+theta⁶/36`. The initial straight Gaussian-core
estimate is Omega≈omega_peak/2=50/s, giving Omega*dt=.375 at nominal dt.
If that rotation were frozen for 1.5 s, the numerical orbit-amplitude factors
would be .85454, .97985 and .99744 for the three timesteps, instead of the
exact value 1. This illustrates how local core motion can be timestep
sensitive while ring-centroid motion is little changed. It is **not** a
prediction of actual ring-core shrinkage: rotation, strain and diffusion
evolve. The actual three-level field contrast is the relevant evidence.

### Full-window finer-timestep result

B2's continuation reaches t=6.225 (step 1660) before its 24-minute segment
wall limit. Both sampled cores have passed x=7 (extents 7.160 and 7.960 R0),
so the complete reference-window comparison is available. The remaining
requested .075 s would not add a required distance interval or a new numerical
contrast; it is not run merely to complete a nominal horizon.

| Common x/R0 interval | Original radius RMS/R0 | Halfdt radius RMS/R0 |
| --- | ---: | ---: |
| .55–1.5 | .02305 | .02624 |
| .55–2.5 | .06071 | .06056 |
| .55–3.5 | .08615 | .08833 |
| .55–5.5 | .16600 | .16739 |
| .55–7.0 | .18845 | .18955 |

Halfdt field-core passages occur at 1.15714, 3.43235 and 5.66786 s. Its
first two-passage cycle is 4.51071 s versus 4.49063 s original, a .45%
difference in the interpolated estimate within the shared field cadence.
Its leading-ring peak-to-peak axial wavelength is 5.280–5.360 R0 versus
3.5465 R0 in LBM, still 48.9–51.1% longer. Thus the improved temporal
resolution does **not** improve the measured full-window LBM trajectory
agreement; the large phase discrepancy is not removed by the tested
refinement. Cadence uncertainty remains explicit rather than being used as
an invented acceptance tolerance.

The [final halfdt section](figures/leapfrogging_study/core_sections/core_sections_leapfrog_Re3000_t6.225_1.png)
still resolves two rounded cores, with peaks .20428/.19024 of initial
vorticity and bridge ratio .07084. Mode-eight axial amplitudes remain below
.00142 R0 and radial amplitudes below .00349 R0; mirror asymmetry=.00736.
This is not evidence of the expected physical breakdown. Final mean
sigma=.22635, divergence=.01514, misalignment=.4222 degrees, CFL=.03054,
and N=16,104 with zero stabilization events. No health stop occurs; survival
is right-censored at 6.225 by the wall limit. Its post-breakdown survival time
is not known.

### Interpretation of the CS diagnosis

The demonstrated findings separate the proposed explanations as follows.

| Explanation | Evidence and status |
| --- | --- |
| Incorrect growth law, viscosity units, or double counting of the initial core | Not demonstrated. The actual CS/LES kernels satisfy their equations, native sigma² follows integrated effective viscosity, and compensated initialization recovers the intended physical Gaussian core. |
| Inappropriate numerical settings | Nominal dt is insufficient to claim converged core fields; two refinements reduce the differences about sevenfold. Initial h refinement changes the sampled field little, but evolved spatial/core convergence remains untested. |
| Finite-core CS modelling/discretization limitation | Strongly supported as a remaining explanation: the spherical diffusion width does not undergo local strain, increasingly dominates represented core width, and directly alters fixed-state self induction. This persists with molecular viscosity alone. Its exact share of the full trajectory error is not uniquely measured. |
| Error elsewhere or reference mismatch | Removing LES worsens early radius error. Saved-state tree source-field errors are small, without proving full transport/stretching convergence. Unbounded VPM versus periodic LBM and absent matched breakdown data remain relevant differences. |

The evidence supports a correctly implemented Gaussian diffusion law with
substantial finite-core representation effects and a separate, demonstrated
timestep error in the nominal core fields. It does not support changing the
factor four, fitting viscosity/Cs, or declaring every remaining discrepancy
an unavoidable physical limitation. There is no demonstrated solver implementation defect in the CS law or its
parameter propagation to correct. The full-window halfdt comparison confirms
that improved temporal resolution does not remove the large spatial phase error.

## Frozen baseline and stabilization comparison

No baseline is frozen, and no stabilization recommendation is justified.
The halfdt control uses the original physical/model parameters, h=.06,
sigma0=.06, Cs=.20, and disabled stabilization, with dt=.00375 and the
verified 120k inactive allocation. The timestep has a convergence-based
justification, but the remaining trajectory/breakdown evidence does not
justify accepting this configuration as the frozen physical baseline.

The following inventory comes from current code inspection, not screening
results. Every row remains untested under a frozen baseline.

| Existing mechanism | Implementation and configuration | Nominal controls | Modified quantity and activation |
| --- | --- | --- | --- |
| Stretching viscosity | `StabilizationOperators.apply_stretching_viscosity`; `StabilizationConfig.stretching_viscosity` | coefficient .5; start step 0; optional existing growth-feedback controls | Adds C h² max(alpha·S alpha / abs(alpha)²,0) to effective viscosity; continuous after the start step |
| Pedrizzetti direction relaxation | `StabilizationOperators.apply_pedrizzetti_relaxation`; `StabilizationConfig.pedrizzetti_relaxation` | factory factor .3 per step, interval 1; magnitude preservation true, moments false | Blends strength toward curl(u); eligible steps only |
| Unnormalized / moment-preserving Pedrizzetti variants | Same operator with `preserve_vortex_strength=False`, optional `preserve_moments=True` | Existing tutorial moment variant uses factor=.384684814725*dt, interval 1 | Allows magnitude change; optional restoration of vector strength and first moments; same schedule |
| Filament refinement | `split_stretched_filaments`; `FilamentRefinementConfig.adaptive` | strength factor 2; offset .25; supplied LES launcher checks every 5 steps and absolute threshold=2*initial peak | Splits eligible particles into two conservative children; positions, strengths and volumes change; inherited sigma unchanged |
| Constrained divergence relaxation | `constrained_divergence_relaxation`; `DivergenceRelaxationConfig.constrained` | regularization .1, max relative correction .02, residual ratio .9; supplied older case interval 25, grid h | Guarded constrained correction to strengths on eligible steps; failed proposals roll back |
| Conservative regularization/remeshing | `regularize`; `regularization_*` fields in `StabilizationConfig` | tail budget .003, energy/enstrophy loss gates .15; requires cadence, grid and particle cap | Redistributes particles and optionally resets sigma when scheduled and a configured health/core/capacity/energy trigger fires |
| Solenoidal remeshing option | Same remeshing implementation, `regularization_solenoidal_remesh=True` | projection trigger .08, max correction .20 | Existing projection variant of remeshing, subject to acceptance gates |

`bounded_domain` deletes particles outside a box. It is not applicable as a
physical stabilizer for these unbounded, translating, closed vortex rings.
The `conservative_filter` convenience factory enables both residual viscosity
and remeshing; its combination must not be mistaken for a single-method control.
Historical combined `p_split`/`p_remesh` cases are likewise excluded.

## Baseline gate and remaining work

No frozen baseline, stabilized comparison, preferred stabilizer, tuning value,
or validated post-breakdown operating range has been established. The
available runs cannot satisfy the requested gate merely by remaining finite:
the continued nominal control loses numerical health with two identifiable
cores, the finer control retains two rounded cores through the reference
window, and the supplied reference does not provide a matched physical
breakdown observation. Screening and tuning remain unstarted, in accordance
with the required ordering. The full definition of done is not met. The recommendation is to retain
these as diagnostic controls, make no CS-growth-law or Cs correction, and
withhold stabilization selection until a physically validated baseline exists.

The required reference clarification is whether the intended target is the
supplied unperturbed Re=3000 merger/trajectory case or the separate seeded
Re=3415 instability case, together with time-resolved reference evidence
through the intended breakdown. Changing Reynolds number and disturbance to
borrow the other case's breakdown would not be a controlled numerical fix.

Only study-relevant uncertainties remain: evolved spatial/core convergence
has not been established by the initial quadrature test; the VPM unbounded
induction and LBM periodic 20R0×7R0×7R0 domain are different; short timestep
controls do not establish convergence of the late numerical health endpoint;
the native energy grid is restart-dependent;
and the fixed-state induction/linear-strain arguments do not uniquely
partition the complete trajectory error. No arbitrary accuracy tolerance has
been introduced.

## Out-of-scope observations

Some historical README links point to absent `docs/reviews/` records. They are
not restored as part of this study.

An existing CS orchestration test fails because its `SimpleNamespace` mock
lacks `vlm_solver` after pre-existing VLM edits. The production baseline has
`vlm_solver=None` and advances normally. This unrelated mock defect is retained.

## Reproduction commands and verification

All study invocations use the Python interpreter listed above and run from
this tutorial directory. Metal required host GPU access; the first sandbox
attempt failed before initialization. No solver source was changed. The
qualified 120k capacity is an allocation for these fixed-population controls,
not an accepted stabilization particle cap or a frozen physical baseline.

```sh
export PYTHONPATH=/Users/flaviomartins/OpenONDA
export MPLCONFIGDIR=/private/tmp/leapfrog_mpl
export XDG_CACHE_HOME=/private/tmp/leapfrog_xdg

# Original, with all supplied numerical/resource settings.
python setup_les.py --variant baseline --case-name study_original_20260913

# Allocation qualification and strict continuation of the same trajectory.
python setup_les.py --variant baseline --particle-capacity 120000 \
  --steps 400 --case-name study_capacity120k_20260913
python solution/leapfrog_accuracy_20260913/provenance/continue_baseline.py
python solution/leapfrog_accuracy_20260913/provenance/continue_baseline_tail.py

# Matched causal controls (execution status is in the ledger).
python setup_les.py --variant baseline --particle-capacity 120000 \
  --smagorinsky 0 --steps 400 --case-name study_cs0_cap120k_20260913
python setup_les.py --variant halfdt --particle-capacity 120000 \
  --steps 800 --case-name study_halfdt_cap120k_20260913
python solution/leapfrog_accuracy_20260913/provenance/quarterdt.py
python solution/leapfrog_accuracy_20260913/provenance/continue_halfdt.py

# Existing user-facing comparison workflow, with explicit run selection.
bash allplot.sh study_original_20260913 study_capacity120k_20260913 \
  study_cs0_cap120k_20260913 study_halfdt_cap120k_20260913 \
  study_quarterdt_cap120k_20260913

# Targeted fixed-state induction diagnostic; does not modify checkpoints.
python assets/check_cs_core_induction.py study_original_20260913 \
  --steps 100 200 400 \
  --output solution/leapfrog_accuracy_20260913/core_induction.json
```

The kernel-law, initial-quadrature and saved-tree-field diagnostic scripts are
archived as `provenance/core_law.py`, `initial_quadrature.py`, and
`tree_operator.py`; their exact result JSON files are adjacent to `provenance`.
The unsuccessful original-allocation LES contrast used the same command as
its retry above, omitting `--particle-capacity` and using case name
`study_cs0_20260913`. It stopped after one step and contributes no physical
comparison. Console logs retain launch/termination reasons. The continuation scripts
name the recorded checkpoints; wall/resource stop positions depend on host
load. No frozen-baseline or recommended-stabilizer command exists because
those stages have not passed the gate.

From the repository root, verification used:

```sh
PYTHONPATH=/Users/flaviomartins/OpenONDA TI_CPU_MAX_NUM_THREADS=2 \
  pytest tests/tutorials/test_vortex_core_agreement.py \
  tests/vpm/test_turbulence_orchestration.py \
  tests/vpm/test_core_spreading_projection.py -q
PYTHONPATH=/Users/flaviomartins/OpenONDA \
  MPLCONFIGDIR=/private/tmp/leapfrog_mpl XDG_CACHE_HOME=/private/tmp/leapfrog_xdg \
  TI_OFFLINE_CACHE_FILE_PATH=/private/tmp/leapfrog_taichi_tests TI_CPU_MAX_NUM_THREADS=2 \
  pytest tests/tutorials/test_vortex_core_agreement.py \
  tests/tutorials/test_vortex_core_sections.py -q
```

The first selection produced 9 passes and the one pre-existing orchestration
mock failure described above. The plotting/diagnostic selection passed all
9 tests, including a known-period crossing test and a tangency/zero-plateau
test. The real CS and Smagorinsky kernels passed their analytical checks.
Logs are retained in `provenance`. Particle checkpoints through t=6 remain
bitwise equal across original and continued controls; all 41 common core-plane VTS files
through t=6 are byte-identical, including the continued segment.

### Changes relative to the starting dirty checkout

The task modifies only these seven files. The pre-existing solver, VLM,
coupler, other-tutorial and test edits were preserved by this task.
`provenance/study_changes.patch` isolates this task's changes from those edits;
`final_receipt.json` records hashes and per-file additions/removals against
the actual starting files, not against a misleading clean-HEAD baseline.

| File | Study change |
| --- | --- |
| `setup_les.py` | Optional inactive particle-capacity override; original default and all physics unchanged |
| `allplot.sh` | Explicit run selection using the existing workflow; actual final samples included for selected runs |
| `assets/assess_lbm_agreement.py` | Bracketed passage/period metrics, a passage-history plot, unambiguous run/Cs/pair labels, and native energy-estimator identification |
| `assets/plot_core_sections.py` | Cs/run labels and optional inclusion of each run's final saved sample |
| `assets/check_cs_core_induction.py` | Targeted fixed-state self/mutual induction and finite-kernel-width diagnostic |
| `tests/tutorials/test_vortex_core_agreement.py` (repository root) | Two generic crossing-diagnostic tests; existing tests preserved |
| `leapfrogging_study.md` | This report and the single authoritative ledger |

No solver-level correction was made because no concrete implementation defect
was established. Shell syntax and scoped whitespace checks pass. The final
plotting/diagnostic test rerun passes all 9 tests. Runtime and LBM-reference
hash checks cover 188 files and show no change; HEAD and branch remain at the
recorded starting values. Native wall times are logged for reproducibility,
not used as performance ratios because other host workloads varied.
