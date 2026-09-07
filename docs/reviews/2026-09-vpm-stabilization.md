# VPM stabilization audit and interaction experiments

**Superseded recommendation:** see [the core-transport investigation](2026-09-vpm-core-transport.md). The baseline below demonstrated runtime, not the requested LBM agreement. The default launcher now tests physics controls.

This audit concerns the classical Gaussian VPM with transposed stretching,
SSPRK3, Core Spreading and Smagorinsky viscosity. No rVPM formulation is used.
The corrected baseline was the longest demonstrated control: it completes 1,200
leapfrog steps, while added stabilization has not established a physically
credible extension through collision breakdown. Natural breakdown remains
unvalidated; the results below distinguish survival from field fidelity.
The experiment data live in `tutorials/vpm/vortex_interactions/study_results/`;
each `result.json` records parameters, a SHA-256 of the source, termination,
and wall time. The source papers establish mechanisms, not a universal safe
parameter setting for this discretization.

## Implementation findings

### Direct answer using radius versus travelled distance

**No two or three stabilization winners have been demonstrated.** The direct
[LBM overlay](../../tutorials/vpm/vortex_interactions/figures/study/lbm_comparison/radius_vs_distance.png)
shows good first-excursion agreement followed by substantial phase drift. The
long VPM runs keep oscillating coherently after the reference curves depart
from that pattern. Survival therefore does not satisfy the requested criterion.

On the same axial interval x/R0=0.55–3.5, the RMS radius discrepancy is 8.95%
of R0 for baseline, 8.94% for splitting, 8.48% for residual viscosity, 9.06% for
normalized realignment, and 8.79% for moment-corrected P. Residual viscosity
reduces that discrepancy by about 5%, a modest improvement rather than a
validated solution. Across x/R0=0.55–7, the three available long trajectories
give 18.63%, 18.35% and 18.86% for baseline, residual viscosity and
moment-corrected P, respectively. Remeshing was cost-stopped too early to
qualify. These are equal-ring, uniform-x comparison scores with no phase
fitting or extrapolation; they are not matched-benchmark error estimates:
the radial initial disturbance and RMS cloud-radius observable differ from
the LBM setup and core-centre observable. The reproducible plot and score
generator is `assets/compare_lbm_trajectory.py` in the tutorial.

### 1. Misalignment was measured against stale vorticity (fixed)

`VPMSolver._update_discretization_health` used the particle `vorticity` array.
It initially contains alpha/V and is subsequently refreshed when a backup is
written. The accepted-step refresh updates velocity and its gradient, but not
that array. Consequently the health check compared current strengths against
old directions and depended on backup cadence. The original baseline stopped
at step 22, t=0.171569, at 25.0598 degrees, despite benign CFL and divergence.

The health check now uses curl(u) from the current velocity gradient, also the
target of P-relaxation. A regression changes the stored vorticity independently
and verifies that only changes to the current curl change misalignment.
Results made before this correction must not be used to rank stabilizers.

### 2. Realignment did not publish its device mutation (fixed)

The Taichi operator changed alpha without incrementing the particle source
revision. Host snapshots populated by the before-event measurement could then
be reused for the after-event measurement, incorrectly reporting zero transfer.
The operator now calls `touch_state()` after its kernel. A CPU Taichi regression
checks both the changed host result and the revision.

### 3. Regularization called an incompatible physical diffusion routine (fixed)

The old regularizer called `grid_based_diffusion`, which is DVH. That routine
requires uniform source cores and a resolved heat-kernel interval. CS+LES
produces unequal cores, and the tutorial's step is too small for the DVH lattice
contract. It also passed a fresh nu*dt despite diffusion already being advanced
by CS, then overwrote the output core radius. This is not a consistent core
reset.

The replacement uses Gaussian convolution: an old core sigma is represented
by particles of core s after spreading their strengths with width
sqrt(sigma^2-s^2). M4' deposition and interpolation between 16 variance bins
approximate this convolution. No physical time or additional nu*dt is added.
Existing moment restoration and event energy/enstrophy gates remain active.
Tail pruning and particle capacity have separate meanings: capacity exhaustion
rejects the proposal instead of silently discarding more circulation.

The field test uses off-lattice sources with unequal cores, checks the field
at independent targets, and checks Gaussian second moments. Bin interpolation,
lattice spacing, pruning, and the event's later core broadening remain numerical
approximations and need convergence checks in the interaction problem. In particular,
the sampled Gaussian convolution does not exactly reproduce its requested
variance when the transfer width is smaller than the grid spacing. The
reported field test covers resolved transfer widths; it is not a universal
accuracy guarantee for arbitrarily frequent core resets.

### 4. Printing impulse could fail at a diagnostic-backend transition (fixed)

The coarse collision splitting run crossed 10,000 particles at step 140. The
accepted diagnostic changed from direct unbounded energy to periodic Fourier
energy. Printing linear impulse then recomputed every quadratic integral with
`record_history=False`; the new energy definition had only one sample, so its
finite-difference rate was undefined and the sampler raised an exception.
This was a diagnostic failure, not evidence of an unstable split cloud.

Linear impulse now uses its direct O(N) particle moment. Trial Fourier integrals
also return a clearly labeled viscous-rate estimate when no accepted derivative
exists, without changing accepted history. A regression covers both directions
of the energy-definition switch, followed by a trial at the same time. The
original failed experiment is retained and the corrected repetition is tagged
`coarse_collision_splitting_fixed`.

### 5. Numerical redistribution transfer was missing from exported budgets (fixed)

Regularization already measured before/after energy and enstrophy, but its
manager retained only event counts and relative strength changes. It now
accumulates the signed energy and enstrophy transfers and includes them in
CSV/restart diagnostics. These quantities cover regularization only: they do
not claim to measure the unrecorded transfer of every realignment or split.
Earlier experiments are explicitly missing this ledger, not assigned zero loss.

### 6. Remeshing energy comparisons mixed definitions (fixed)

A remap crossing 10,000 particles could compare unbounded direct energy against
periodic Fourier energy. Even above that threshold, source and target grids
could differ. Transfer acceptance now uses one padded grid covering both
clouds, one periodic quadratic form, and a checked sixth-order core expansion.
The same grid is retained through moment correction and core-broadening trials.

### 7. Gaussian FFT filtering left Nyquist aliases unsmoothed (fixed)

The Fourier-integral wave-number helper set each Nyquist frequency to zero.
That convention belongs to certain real-grid first derivatives; applying it to
the Gaussian exponent makes high-frequency aliases look like zero-frequency
content. A sparse two-source remapping test exposed a false 47% energy change
although its independent reconstructed vorticity error was only 0.177%.
Physical FFT wave numbers restore consistent filtering. The same remapping
now passes a 0.5% bound for both quadratic integrals, and a single Gaussian's
periodic enstrophy matches its analytical integral to 1e-7 relative tolerance.

Pre-fix FFT energy/enstrophy values must not rank strategies. Dynamics without
energy-dependent stabilization remain usable; their saved fields can be
remeasured. Remeshing trials using the former acceptance calculation are
superseded by the `qualified_` runs. The constrained real-grid Helmholtz
projection retains its separate derivative convention.

## Correspondence with publications

### Initial disturbance support (optional correction for refinement studies)

The original support is a circular torus, while its vorticity distribution is
radially disturbed. Truncation therefore removes different tails at different
azimuths. Measured radial centroid-mode amplitudes at h=0.04, 0.035, 0.03 were
0.0331, 0.0379, 0.0403 R0 for a prescribed 0.05 R0 disturbance. This changes the
effective initial perturbation during a spatial-refinement study.

`study.py --support disturbed` moves the toroidal geometry with the centreline
and applies the cylindrical Jacobian to volume weights. The corresponding
mode measurements are 0.0473, 0.0483, 0.0476 R0; the residual difference from
0.05 includes finite-bin and tube-centroid effects. A regression checks the
coordinate map, volume Jacobian, and unchanged core radii. Original-support
screening runs remain separately identified in their metadata.

| Method | What the implementation actually does | Interpretation |
|---|---|---|
| Transposed stretching | J^T alpha, advanced with coupled position/strength SSPRK3 | Corresponds to the transposed form in Winckelmans & Leonard. Exact pair cancellation assumes matching pair kernels; unequal CS cores and tree approximation require measured conservation checks. |
| Smagorinsky | nu_t = (Cs Delta)^2 sqrt(2 S:S), Delta=V^(1/3) | Algebraically the equilibrium Smagorinsky model. Applying local nu_t through core growth is a VPM model adaptation, not the complete variable-coefficient vorticity SGS operator of Mansfield et al. Splitting halves V and reduces Delta, so it also changes modeled dissipation. |
| Residual stretching viscosity | C V^(2/3) max(alpha.S.alpha/‖alpha‖^2,0) | Dimensionally consistent, positive added viscosity. A heuristic closure in this code; no original publication establishing this precise formula or C=0.5 was found. It must be labeled and measured as added dissipation. |
| P-relaxation | alpha_new=(1-f dt)alpha + f dt ‖alpha‖ curl(u)/‖curl(u)‖ | The unnormalized blend matches Winckelmans (1995), equation 11, adapting Pedrizzetti's singular method to regularized blobs. |
| Normalized realignment | Renormalizes the blend to the original ‖alpha‖ | An additional code adaptation; preserving each magnitude does not preserve total vector strength, impulse, energy or helicity. The original tutorial uses a large 0.3 blend every step. |
| Moment-corrected P-relaxation | Applies a minimum-norm nine-moment correction after the blend | An explicit adaptation. It preserves global linear constraints; it is not an orthogonal Helmholtz projection or proof of local physical accuracy. |
| Filament splitting | Two half-strength, half-volume children displaced along alpha, unchanged sigma | Conserves vector strength and linear/kernel-corrected angular impulse algebraically. It refines material-line sampling; it does not sharpen the smoothing kernel or restore transverse resolution. It is not Rossi-style core-size splitting. |
| Constrained divergence relaxation | Regularized particle-mesh Helmholtz correction plus moment and quadratic constraints | Motivated by W-relaxation (equation 12), but its regularization, constraints, Fourier boundary treatment and acceptance gates are additional choices. It assumes nearly equal Gaussian widths and can reject a CS+LES cloud. |
| Gaussian core remeshing | Rebuilds a lattice while retaining the old Gaussian width in the represented field | Consistent with the convolution identity and the remeshing/overlap rationale. The finite-bin implementation and conservation gates are adaptations, not a verbatim published algorithm. |

The regularized P-relaxation formula and W-relaxation interpolation equation
are directly visible on page 394 of [Winckelmans' 1995 CTR research brief](https://ntrs.nasa.gov/api/citations/19960022324/downloads/19960022324.pdf).
That report also cautions that numerical redistribution/relaxation dissipation
should remain below modeled LES dissipation. Frequency f has units of inverse
time; keeping a fixed blend when halving dt doubles the relaxation rate.

Primary references:

- [Pedrizzetti (1992), Insight into singular vortex flows](https://doi.org/10.1016/0169-5983(92)90011-K).
- [Winckelmans & Leonard (1993), Contributions to vortex particle methods](https://doi.org/10.1006/jcph.1993.1216).
- [Cottet & Koumoutsakos (2000), Lagrangian grid distortions](https://www.cambridge.org/core/books/vortex-methods/lagrangian-grid-distortions-problems-and-solutions/85E1D2B2C07B14732F617653D7CA65E1).
- [Mansfield, Knio & Meneveau (1998), A dynamic LES scheme for the vorticity transport equation](https://www.sciencedirect.com/science/article/pii/S002199919896051X).
- [Mansfield, Knio & Meneveau (1999), Dynamic LES of colliding vortex rings using a 3D vortex method](https://doi.org/10.1006/jcph.1999.6258).
- [Rossi (1996), Resurrecting core spreading vortex methods](https://doi.org/10.1137/S1064827593254397).
- [Cheng, Lou & Lim (2015), Leapfrogging of multiple coaxial viscous vortex rings](https://doi.org/10.1063/1.4915890).
- [McKeown et al. (2018), Cascade leading to the emergence of small structures in vortex ring collisions](https://doi.org/10.1103/PhysRevFluids.3.124702).

## Physical interpretation rules

1. A diagnostic stop is not a numerical blow-up. Keep the same thresholds for
   matched comparisons and retain the rejected/terminal sample.
2. Compare at common physical times, not each method's different endpoint.
   Report numerical survival and physics error separately.
3. Energy need not be constant in this viscous LES. An excessive extra loss is
   as problematic as unexplained energy growth. Enstrophy may rise during
   stretching and physical breakdown; that alone is not numerical instability.
4. Global impulse conservation is necessary evidence, not sufficient validation.
   The sampled angle and nearest-neighbor overlap are also incomplete measures:
   clustering can hide a hole in a different direction.
5. The tutorial's `tube_circulation` is sum‖alpha‖/(2 pi R_rms), a geometric
   proxy valid for coherent rings. It is not a material circulation integral
   after folding, reconnection or breakdown. Remeshed group labels are not
   passive material tracers either.
6. The initial mode-eight amplitude is 0.05 R0, not an infinitesimal perturbation.
   Radial and axial disturbances are different initial-value problems. The
   Cheng paper describes an axial perturbation in its Fig. 3 validation example
   at Re=3415, then explicitly excludes instability from its parametric study.
   The digitized Fig. 5 reference is unperturbed. Matching Re, radius and
   separation alone does not prove benchmark equivalence. Its curves are a kinematic check, not
   proof of correct instability growth or late turbulent dynamics.
7. Complete leapfrogging does not mean an arbitrary number of passes: viscosity
   and instability determine when coherent rings disappear. Collision can
   involve elliptic and Crow mechanisms in addition to a seeded ring mode.
   Assess the structures and mode growth rather than demanding a preset movie.

## Results

### Recommendation and completed demonstration

Keep LES + transposed stretching + SSPRK3 as the default. The original-support
baseline completed 1,200 steps after the health correction, compared with its
former stop at step 22. The final implementation with disturbance-following
support also completed 1,200 steps, reaching t*=29.4 with 17,544 particles in
32.1 minutes of recorded wall time. Health thresholds were unchanged.

The original-support trajectory contains four centroid overtakes (two full
leapfrog cycles, counting a cycle as restoration of the initial ring order), near t*=3.55, 10.54, 17.64 and 24.62. These are
kinematic events, not proof of correct instability growth or reconnection.
Reconstructed vorticity still shows two broadening coherent tubes at the end.
**Natural turbulent breakdown has not been demonstrated or validated.**

The added methods did not establish a useful physical survival advantage for
collision at the tested spacings. The default `allrun.sh` therefore runs the
baseline controls. `--campaign strategies` retains baseline plus splitting,
remeshing and weak P + remeshing as explicit comparisons; `--campaign screen`
adds the remaining methods and experimental combinations. None is silently
promoted to a validated breakdown strategy.

### Matched collision screening

These rows share Re=3000, h=0.04, sigma0=0.08, disturbance-following support,
mode eight with amplitude 0.05, Cs=0.20, dt=0.0077985922, and the same health
limits. The intended horizon was 400 steps except the coarser-grid solenoidal
trial (300). `resolution_lost` is a divergence-limit stop; `failed` means the
stabilization proposal was rejected. The latter is not a flow blow-up.

| Method                                   |   Accepted steps |   Final t* |   Last sampled N |   Wall min | Termination     |
|:-----------------------------------------|-----------------:|-----------:|-----------------:|-----------:|:----------------|
| Baseline                                 |              179 |      4.386 |             6688 |      1.901 | resolution_lost |
| Filament splitting                       |              180 |      4.410 |            16224 |      3.329 | resolution_lost |
| Residual viscosity C=0.5                 |              180 |      4.410 |             6688 |      2.466 | resolution_lost |
| Normalized blend 0.3                     |              189 |      4.631 |             6688 |      1.853 | resolution_lost |
| Weak moment-corrected P                  |              178 |      4.361 |             6688 |      1.798 | resolution_lost |
| Gaussian remeshing                       |              167 |      4.092 |            29578 |      8.552 | resolution_lost |
| Weak P + remeshing                       |              174 |      4.263 |            29529 |      7.010 | resolution_lost |
| Weak P + splitting + remeshing           |              183 |      4.484 |            43670 |     13.229 | resolution_lost |
| Frequent remesh + constrained projection |              159 |      3.896 |            52026 |     13.981 | resolution_lost |
| Constrained divergence relaxation        |               24 |      0.588 |             6688 |      2.002 | failed          |
| Solenoidal remesh, coarser grid          |              119 |      2.916 |            39730 |      9.108 | failed          |

Wall times include initialization and diagnostics. Several jobs shared the
same Metal GPU during exploration, so these are observed campaign costs, not
isolated performance benchmarks. Particle growth and the rejection reasons
are firmer evidence of cost than small differences between wall times.

The 189-step normalized run is only 5.6% longer than the 179-step control.
Splitting and residual viscosity each add one step; weak P alone loses one.
The combined weak-P/split/remesh method gains four steps while growing the
cloud to 43,670 particles. These differences do not justify claiming a
reliable extension through collision breakdown.

Checking remeshing only every 50 steps missed a rapid increase: divergence
rose from 0.051 at step 160 to 0.120 at step 174, before the next check. Shorter
checks are necessary when that growth begins, but not sufficient: the tested
frequent constrained-projection schedule stopped at 159 with 52,026 particles.
An even earlier projection was rejected because combined enstrophy loss
exceeded the unchanged 5% event limit.

The solenoidal-grid profile accepted one remap but its next proposal required
159,712 particles, exceeding the 120,000 cap. Tighter-tail and earlier-grid
variants were also capacity-limited or impractical. An early variant created
102,289 particles at step 10 and was stopped as a cost screen; this was not a
successful long-running simulation.

### Field fidelity distinguishes the leapfrog methods

At step 400 (t*=9.8), direct Gaussian Biot–Savart evaluation at the same 1,024
targets around the original-support baseline gives these relative velocity
differences. The baseline is a control, not an independently validated truth.

| Added method | Velocity difference from baseline | Particles at step 400 |
|---|---:|---:|
| Filament splitting | 0.0142% | 17,900 |
| Normalized blend 0.3 each step | 35.39% | 17,544 |
| Unnormalized, moment-corrected blend 0.03 | 10.12% | 17,544 |

Splitting is the least intrusive tested alternative over this interval, but
has not demonstrated longer collision survival. It leaves sigma unchanged,
so it cannot be a remedy for lost transverse/core resolution.

The moment-corrected 0.03 run also completed 1,200 steps. At step 300, however,
its second ring's radial/axial mode-eight centroid amplitudes were
0.00348/0.00134 R0 versus 0.01719/0.01386 R0 in the baseline. It strongly
suppresses the measured seed dynamics. The common-box energy ratios at
step 1,200 were nevertheless very close: 0.68057 versus 0.68223. Conserved
moments and similar total energy therefore do not establish local fidelity.

Gaussian remeshing increased the h=0.035 leapfrog cloud from 17,544 to 55,138
particles at its first event. That screen was stopped at accepted step 153;
a coarse remesh screen reached 75,182 particles at step 200 and was likewise
stopped for cost. Neither had demonstrated a survival need in a case already
completed by the baseline. They are retained as interrupted screens, not
ranked as successful extended trajectories.

### Resolution and diagnostic controls

At t*=2.45, direct unbounded velocities were compared on 1,024 identical
random targets in [-0.75,0.75] x [-2,2] x [-2,2], with seed 1729.

- Halving dt at h=0.04 changes sampled velocity by 1.129%.
- Changing h=0.04 to 0.035 at fixed Cs changes it by 10.409%.
- Changing h=0.035 to 0.03 at fixed Cs changes it by 10.202%
  (normalized by the h=0.03 field).
- Matching nominal Cs*h reduces those spacing differences only to 9.769%
  and 9.529%, respectively (each normalized by its coarser control).
  This is a nominal LES-filter control; local V^(1/3) is not exactly h.
- Initial h=0.035 and h=0.04 fields differ from h=0.03 by only 0.580% and
  1.003%, respectively, on the same target set.

These are sampled differences, not global error bounds or a proven convergence
order. The refinements do not yet certify breakdown physics. Besides the LES
filter change, Gaussian-tail truncation and quadrature leave small initial
profile differences even after matching the disturbance support.

The saved original baseline was also remeasured with corrected Gaussian FFT
filtering on one common box. Its final E/E0 was 0.68223 at audit spacing 0.05,
0.68217 at spacing 0.04, and 0.68305 with increased padding. The core-expansion
order difference was below 1e-7 for these audits. These checks support the
energy measurement, not convergence of the simulated turbulence.

The final disturbance-following baseline has E/E0=0.68909 and Z/Z0=0.16032
at step 1,200 on its common audit box (spacing 0.05). These ratios belong to
that initial layout and should not be substituted for the original-support
comparison above.

### Practical strategy

1. Use the corrected baseline and retain the unchanged health gates. Validate
   the actual Gaussian field and the seeded modes before optimizing survival.
2. Use factor-two filament splitting when material-line sampling stretches;
   measure particle cost and remember that splitting changes the current
   volume-based LES filter. The tested early field perturbation is small.
3. Treat remeshing as a measured representation change: reset Gaussian variance
   without another diffusion step, restore moments, audit the same quadratic
   form on both sides, and retain signed transfer budgets. Do not discard
   additional strength merely to fit a particle cap.
4. If experimenting with realignment, parameterize a physical frequency and
   scale the blend with dt. The tested weak frequency is 0.384685 /s. Avoid
   promoting a small angle to evidence of correct dynamics; neither weak nor
   strong realignment qualified a collision extension here.
5. For complete breakdown, first obtain matched initial fields and adequate
   transverse resolution. The current results motivate a more efficient
   remeshed particle-mesh or multiresolution approach, and/or higher-order
   redistribution, rather than escalating artificial viscosity or relaxing
   health limits. Those architectural changes remain future work; they are
   not implemented or validated by this campaign.

### Reproducibility and checks

The run table links to local result directories through the `Run` column in
`figures/study/collision_results.csv`. Each result retains its configuration,
source fingerprint, status and wall time; snapshots support independent field
checks. Source evolved during the audit: pre-fix remeshing and FFT results are
superseded as explained above. The `qualified_` prefix denotes the corrected
implementation campaign, not a certificate of physical validation.

The revised launcher was exercised end to end with `allrun.sh --quick`: its
leapfrog baseline reached 400 steps and its collision baseline stopped at 179,
then both result summaries and plots were generated. The focused suite passed
84 tests, including cache publication, current-curl health, Gaussian field and
Nyquist checks, projection, restart transfer ledgers and launcher behavior.
Three additional numerical-qualification tests passed. Taichi emitted
non-fatal cache-lock/deprecation warnings; no test failure remained.

Visual evidence:

- [Final long-run particle animation](../../tutorials/vpm/vortex_interactions/figures/study/qualified_leapfrog_baseline.gif).
- [Original-control reconstructed vorticity](../../tutorials/vpm/vortex_interactions/figures/study/leapfrog_baseline_vorticity.png).
- [Final-layout reconstructed vorticity](../../tutorials/vpm/vortex_interactions/figures/study/qualified_leapfrog_baseline_vorticity.png).
- [Collision reconstructed vorticity](../../tutorials/vpm/vortex_interactions/figures/study/control_collision_h035_vorticity.png).
- [Recorded campaign summary](../../tutorials/vpm/vortex_interactions/figures/study/summary.md).


## Additional classical method: solenoidal remeshing

`solenoidal_remeshing` projects the redistributed Cartesian strength field with
`P(k)=I-kk^T/|k|^2` before pruning. Gaussian convolution commutes with this
projection, and `k cross P(k)alpha = k cross alpha`, so the complete-grid
velocity is retained while longitudinal vorticity is removed. We add padding
before the projection, then apply the existing tail budget, moment correction,
and common-grid energy/enstrophy acceptance checks. Padding, pruning and moment
restoration mean that particle-field solenoidality and unbounded velocity
preservation are approximate and must be measured.

This is motivated by [van Rees, Leonard, Pullin & Koumoutsakos (2011), section
2.1](https://doi.org/10.1016/j.jcp.2010.11.031), which combines remeshing with
spectral solenoidal reprojection every 5–10 steps. Their method is a periodic
hybrid particle-mesh solver; our padded, localized Gaussian reconstruction with
treecode evolution is an adaptation. It does not implement their whole solver.
It is unrelated to the reformulated VPM excluded from this study.

An independent initial-field check at 256 fixed targets gave 0.372% relative
unbounded velocity error after projected remeshing with grid spacing 0.05,
core 0.08 and tail budget 0.02 (6,620 output particles). Unprojected remeshing
with those settings gave 1.322% (4,178 particles). This is a one-event field
check, not a long-time validation. Delaying projection to step 80 instead
required 122,760 particles at spacing 0.04 and tail budget 0.01, exceeding the
120,000-particle budget. Early projection and a coarser remeshing grid are
therefore tested separately rather than silently raising the capacity.
