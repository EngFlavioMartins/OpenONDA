# Vortex-particle large-eddy simulation: research history and validation plan

## The project in one paragraph

OpenONDA needs a turbulence model for three-dimensional vortex-particle
simulations that cannot resolve every turbulent scale. This is the purpose of
large-eddy simulation (LES): compute the large, resolved motions and model the
effect of the smaller, unresolved motions. The present candidate reconstructs
some missing small-scale velocity and calculates its effect as a vorticity
source, or **torque**. The mathematics and periodic-grid implementation are
encouraging, but the closure has not yet run inside the VPM solver. It must
therefore be called a candidate, not a validated particle LES model.

## Why a new model is needed

The incompressible vorticity equation is

$$
\frac{\partial\boldsymbol\omega}{\partial t}
+\boldsymbol u\cdot\nabla\boldsymbol\omega
=\boldsymbol\omega\cdot\nabla\boldsymbol u
+\nu\nabla^2\boldsymbol\omega.
$$

Here $\boldsymbol u$ is velocity, $\boldsymbol\omega$ is vorticity, and $\nu$
is molecular viscosity. LES applies a spatial filter because the numerical
method cannot represent motions smaller than its working resolution. Filtering
the nonlinear terms creates an unknown subgrid-scale contribution:

$$
\frac{\partial\overline{\boldsymbol\omega}}{\partial t}
+\overline{\boldsymbol u}\cdot\nabla\overline{\boldsymbol\omega}
=\overline{\boldsymbol\omega}\cdot\nabla\overline{\boldsymbol u}
+\nu\nabla^2\overline{\boldsymbol\omega}
+\boldsymbol g_{SGS}.
$$

$\boldsymbol g_{SGS}$ is the influence of unresolved turbulence on resolved
vorticity. This is the quantity the project must model.

OpenONDA represents vorticity with particles. Molecular viscosity will be
handled separately by Grid-Based Diffusion (GBD). The LES model must therefore
add $\boldsymbol g_{SGS}$ directly to particle circulation—the amount of
vorticity carried by each particle. It must not disguise the turbulent
contribution as molecular viscosity or as a change in particle core radius.

## How the research reached the present candidate

### 1. The filter scale was separated from particle core size

The Mansfield formulation was audited from the primary paper. The important
lesson was that the LES filter describes **resolved numerical scale**, whereas
particle core radius describes the particle basis used to represent a field.
They are not interchangeable. In this project the LES width is tied to the
working grid spacing $h$, not to particle core radius $\sigma$.

This removed an early architectural confusion. Core Spreading is not required;
OpenONDA can use GBD for molecular diffusion while the LES torque is handled
separately.

### 2. Simple coefficient models were rejected

The first modeling route was Mansfield's dynamic eddy-diffusivity idea,
together with the Germano–Lilly procedure for determining a coefficient from
the evolving flow. The motivation was attractive: let the simulation determine
how much small-scale dissipation it needs instead of prescribing a constant.

Tests showed that one coefficient, and later two functional basis terms, could
not represent unresolved vorticity transport and unresolved vortex stretching
simultaneously. Even when the coefficients were chosen optimally using the
reference answer, typical field correlations remained around $0.4$. A mixed
model combining reconstruction with another dynamically fitted coefficient was
also unreliable. These approaches were stopped rather than tuned further.

### 3. The project moved from a functional model to a structural model

A functional model prescribes the effect that unresolved turbulence *should*
have, usually as extra dissipation. A structural model instead attempts to
reconstruct the missing field and recompute the missing nonlinear products.

A preliminary van-Cittert approximate-deconvolution test produced a clear jump
in accuracy. This justified studying Yuan et al.'s dynamic iterative
approximate deconvolution (DIAD), which constructs the reconstruction weights
from the resolved field. If $\boldsymbol u^\star$ is the reconstructed
velocity, the structural stress is

$$
\tau^S_{ij}
=G_\delta\!\left(u_i^\star u_j^\star\right)
-G_\delta\!\left(u_i^\star\right)
 G_\delta\!\left(u_j^\star\right).
$$

The associated vorticity torque is

$$
\boldsymbol g_{SGS}
=-\nabla\times\left(\nabla\cdot\boldsymbol\tau^S\right).
$$

This construction follows directly from the exact filtered equation instead
of guessing a new torque shape. It produced a large improvement over the
rejected coefficient models.

### 4. A dissipative correction remains a hypothesis

Yuan also includes a small-scale eddy-viscosity (SSEV) correction, denoted here
by $\boldsymbol\tau^{SSEV}$. Applying it continuously removed too much energy
in some development cases. The current candidate therefore uses

$$
\boldsymbol\tau^m
=\boldsymbol\tau^S+s\boldsymbol\tau^{SSEV},
\qquad
s=\max\left(0,1-\frac{\varepsilon}{e}\right),
\qquad
\varepsilon=0.01.
$$

$e$ measures how poorly the reconstructed field reproduces the known filtered
field. The correction is activated only when this inconsistency is larger than
$1\%$. This activation rule was created in this project; it is **not** claimed
as a result of Yuan et al. The structural model has evidence behind it, but the
activation rule is still under test.

## Frozen model definition

The filter and reconstruction are now fixed so later tests cannot be improved
by case-specific tuning:

- Gaussian filter:

  $$
  G_\delta(\boldsymbol k)
  =\exp\left(-\frac{\delta^2|\boldsymbol k|^2}{4}\right).
  $$

  Here $\boldsymbol k$ is the wavenumber vector: larger
  $|\boldsymbol k|$ represents smaller spatial motion.

- Yuan filter width: $\Delta=\sqrt6\,\delta$.
- Numerical hierarchy: $\Delta=2h$.
- DIAD reconstruction: five-point stencil in each direction, two weight
  updates, and weights constrained to sum to one so a uniform velocity remains
  unchanged.
- Nearly singular systems: one fixed singular-value-decomposition cutoff is
  used to discard numerically unresolved directions. The threshold is identical
  for every flow; there is no case-specific regularization.
- Molecular diffusion: GBD receives molecular $\nu$ only.
- LES action: the modeled torque is added explicitly to vorticity circulation.

Changing any of these choices resets the current time-dependent validation and
all later gates.

## How to read the reported measurements

- **Correlation:** whether modeled and exact torque have the same local
  pattern. One is perfect; zero means no linear agreement.
- **Relative $L_2$ error:** total field error divided by the size of the exact
  field. Zero is perfect.
- **Transfer ratio:** modeled mean enstrophy transfer divided by the exact
  transfer. One is correct; values above one are too strong.
- **Shell-transfer error:** error in how transfer is distributed over
  wavenumbers, hence over large and small resolved scales. Zero is perfect.
- **Energy spectrum:** kinetic energy carried by each wavenumber. Excess energy
  near the largest represented wavenumbers signals unresolved pile-up.
- **Enstrophy:** one half of mean squared vorticity. It is more sensitive than
  energy to small-scale errors.
- **Condition number:** sensitivity of the reconstruction weights to numerical
  error. Very large values require precision and repeatability checks.

## Evidence obtained so far

### What the evidence can honestly support

The different tests answer different questions. They should not be combined
into one vague statement that “the model works.”

| Claim | Present evidence | Assessment |
|---|---|---|
| The filtered equation, signs, and dimensions are correct | Independent derivation and exact algebraic identities | strong |
| The structural model approximates an exact missing torque on periodic fields | A-priori AGARD and homogeneous-turbulence tests | encouraging |
| The model can improve one evolving periodic LES | One long $64^3/32^3$ paired calculation | preliminary; one seed and low Reynolds number |
| The underlying vortex-particle solver can reproduce a known flow | Raw vortex-ring VPM trajectory compared with Saffman theory | supported for that laminar ring |
| A known torque can be transferred to particle circulation correctly | Manufactured-source refinement and invariant test | supported for smooth, volume-preserving particle deformation |
| Structural DIAD works inside VPM | No completed flow calculation | **not demonstrated** |
| The model works with bodies, inflow, or realistic turbulence | No completed calculation | **not demonstrated** |

The periodic reference and LES share Fourier operators, filtering, and domain
assumptions. That makes their comparison precise, but also creates a risk of
common numerical bias. It cannot replace a particle calculation or an
independent physical benchmark.

### Tests on known turbulence fields

These are *a-priori* tests: an accurately resolved field is filtered, its exact
missing torque is computed, and the model is evaluated without advancing time.
AGARD refers to the former Advisory Group for Aerospace Research and
Development, which published the reference turbulence dataset.

| Field | Cases | Correlation | Relative $L_2$ error | Transfer ratio | Shell error |
|---|---:|---:|---:|---:|---:|
| AGARD decaying homogeneous turbulence, development data | 2 | 0.819 | 0.580 | 0.726 | 0.343 |
| Stationary forced homogeneous turbulence, development data | 18 | 0.989 | 0.149 | 0.974 | 0.040 |
| Transient homogeneous turbulence, untouched holdout data | 15 | 0.961 | 0.275 | 0.891 | 0.189 |

The untouched holdout result is important: it was not used to invent the
activation rule. Nevertheless, these fields are small and mostly
homogeneous isotropic turbulence (HIT) cases, so they are not publication-level
validation by themselves.

[A-priori comparison figure](figures/vpm_les/stage_4a_apriori_metrics.png) ·
[formulation residuals](figures/vpm_les/stage_4a_formulation_residuals.png)

### Exact time-dependent solutions

Before judging turbulence physics, the research solver was tested against a
decaying shear wave and an Arnold–Beltrami–Childress (ABC) field. Their exact
solutions are

$$
\boldsymbol u(t)=\boldsymbol u_0e^{-\nu k^2t},
\qquad
E(t)=E_0e^{-2\nu k^2t},
\qquad
Z(t)=Z_0e^{-2\nu k^2t}.
$$

For these fields the exact subgrid torque is zero. The numerical histories
overlap the theoretical curves, the measured time order is at least $2.006$,
and the largest false modeled torque is $2.37\times10^{-14}$. Taylor–Green
initial energy, enstrophy, and energy-decay rate agree with theory within
$1.23\times10^{-15}$.

This verifies the signs, normalization, filtering, divergence control, and
time integration. It does not yet prove that the model represents turbulence.

[Exact-solution overlays](figures/vpm_les/stage_4b0_exact_overlays.png) ·
[time-step convergence](figures/vpm_les/stage_4b0_temporal_convergence.png) ·
[Taylor–Green identities](figures/vpm_les/stage_4b0_tgv_references.png)

### Identical forcing on the reference and LES grids

Before running forced turbulence, the external stirring method was verified.
One smooth random acceleration history is prescribed for the resolved LES
equation. The reference-grid force is then chosen so that filtering it produces
exactly that LES force:

$$
G_\delta\boldsymbol f_{reference}=\boldsymbol f_{LES}.
$$

This matters because applying the same *raw* force on both grids would not be
consistent: the reference force is filtered before it appears in the LES
equation.

The construction follows Eswaran and Pope's primary forcing study. They use a
low-wavenumber Ornstein--Uhlenbeck acceleration, project it onto the
divergence-free plane, and prescribe the temporal covariance

$$
\left\langle b_i(t)b_j^*(t+s)\right\rangle
=2\sigma_f^2\delta_{ij}\exp(-s/T_f).
$$

This is preferable here to forcing proportional to the instantaneous velocity:
it remains an external field, independent of which LES closure is being
tested. Consequently every model can receive the same filtered physical force.

The forcing follows a prescribed temporal correlation and is divergence free,
nearly isotropic, and restricted to wavenumbers $1\leq|\boldsymbol k|\leq2$.
The filtered reference-grid and LES-grid fields agree to
$3.16\times10^{-16}$. The
component-variance departure from perfect isotropy is $0.0105$, and the
measured correlation curve differs from its theoretical curve by $0.025$ in
root-mean-square terms.

Without a subgrid model, the exact resolved energy balance is

$$
\frac{dE}{dt}=P_f-2\nu Z,
$$

where $P_f$ is power supplied by the forcing. The numerical balance converges
with measured order $1.978$; its finest-step relative residual is
$9.85\times10^{-6}$. The forcing method therefore passes and is now frozen for
paired comparisons.

[Forcing/reference overlays](figures/vpm_les/stage_4b1_forcing_verification.png) ·
[energy-balance overlay](figures/vpm_les/stage_4b1_forcing_budget.png)

The primary study reports that this flow usually needs about three to five
large-eddy turnover times to become statistically stationary. It also warns
that reliable averages require a sampling interval much longer than the
quantity's correlation time. The present protocol therefore discards the first
five turnover times and judges the next ten. Differences between the first and
second half of that window must be smaller than their autocorrelation-based
95% uncertainty and smaller than a 20% absolute guardrail.

### First time-dependent turbulence pilot

A reduced $48^3/24^3$ Taylor–Green calculation was run to $t=4$. Without a
subgrid model, final energy error was $12.4\%$ and enstrophy error was
$37.2\%$. Structural DIAD reduced them to $0.25\%$ and $3.35\%$.

This is encouraging, but it also exposed an unresolved issue. The proposed
sensor switched the dissipative correction off after $t=0.2$, making the
sensed model identical to the structural model. The fully active correction
gave better final enstrophy but slightly worse integrated energy and spectral
error. The sensor is therefore neither accepted nor rejected yet.

The constrained reconstruction systems also reached condition numbers near
$10^{17}$. The fixed truncated solve remained stable, but precision sensitivity
must be tested explicitly.

[Time histories](figures/vpm_les/stage_4b_pilot_histories.png) ·
[energy spectra](figures/vpm_les/stage_4b_pilot_spectra.png) ·
[model diagnostics](figures/vpm_les/stage_4b_pilot_diagnostics.png)

### First forced-turbulence pilot with a resolved reference

An initial $48^3/24^3$ comparison was rejected because the proposed reference
contained as much as $7.0\%$ of its energy near its own resolution limit. It
would have been misleading to call that field a reliable reference.

The pilot was repeated at $64^3/32^3$ and lower Reynolds number
($\nu=0.02$). The fine-grid high-wavenumber energy then remained below
$0.476\%$, satisfying the predeclared $1\%$ reference-resolution limit. All
model energy balances closed within $7.0\times10^{-4}$.

Without a subgrid model, final energy and enstrophy errors were $5.57\%$ and
$26.1\%$. Structural DIAD reduced them to $0.80\%$ and $1.23\%$. Its mean
energy-spectrum error was $1.94\%$, compared with $9.16\%$ without a model.
The continuously active eddy-viscosity correction gave a slightly smaller
mean spectrum error, $1.86\%$, but larger final energy and enstrophy errors,
$1.63\%$ and $1.99\%$.

The sensor remained off throughout, so sensed DIAD was identical to structural
DIAD. This pilot therefore supports the structural closure but does not decide
whether the dissipative correction should be sensed or continuously active.
The run spans only about $0.35$--$0.38$ large-eddy turnover times and has
Taylor-scale Reynolds number near $5.8$; it is a numerical and physical screen,
not turbulence-model qualification.

[Forced-flow histories](figures/vpm_les/stage_4b1_forced_hit_histories.png) ·
[spectrum/reference overlay](figures/vpm_les/stage_4b1_forced_hit_spectra.png) ·
[theoretical energy balances](figures/vpm_les/stage_4b1_forced_hit_budgets.png) ·
[reference-resolution check](figures/vpm_les/stage_4b1_forced_hit_reference_resolution.png)

### Statistically stationary reference gate

A first $64^3$ reference was stopped after 7.73 turnover times. It was well
resolved and its mean power balance was correct, but a three-turnover slice was
too short to distinguish slow random fluctuations from drift. That attempt is
retained as a failed result; it was not reclassified.

The test was repeated for 15.44 turnover times. The first five were discarded,
and the final 9.98-turnover window covered 12.42 measured correlation times.
All checks passed:

- energy and dissipation slopes were $0.0073$ and $0.0166$ per turnover;
- mean power input and dissipation differed by $1.81\%$;
- time-averaged component-energy anisotropy was $5.98\%$;
- $k_{max}\eta$ remained above $2.095$ compared with the theoretical minimum
  of one;
- the high-wavenumber energy fraction remained below
  $2.35\times10^{-5}$ compared with the $0.01$ limit.

The faster reference implementation uses

$$
\frac{\partial\boldsymbol\omega}{\partial t}
=\nabla\times(\boldsymbol u\times\boldsymbol\omega)
+\nu\nabla^2\boldsymbol\omega+\nabla\times\boldsymbol f.
$$

It agrees with the previously verified convection-plus-stretching form to
$9.38\times10^{-16}$, so this is a computational optimization rather than a
change of equations.

[Stationarity and theoretical limits](figures/vpm_les/stage_4b2_stationary_reference.png) ·
[final spectrum and $k^{-5/3}$ slope guide](figures/vpm_les/stage_4b2_stationary_spectrum.png)

### First stationary comparison of the LES models

The qualified forcing protocol was then used for one continuous
$64^3$ reference and four $32^3$ calculations: no subgrid model, structural
DIAD, structural DIAD with the dissipative correction always active, and the
proposed sensed correction. The calculation covered $15.49$ large-eddy
turnover times. The first five were excluded and the final $9.90$ were used for
comparison. The reference independently passed the same stationarity and
resolution checks as the previous qualification run.

Structural DIAD improved every predeclared physical comparison. Relative to
the filtered reference, its mean energy error was $2.64\%$, mean enstrophy
error was $1.81\%$, and time-mean spectrum error was $4.62\%$. Without a
subgrid model these errors were $3.49\%$, $7.40\%$, and $7.56\%$. Thus the most
small-scale-sensitive error, enstrophy, fell by about $76\%$, while the
spectrum error fell by $39\%$. Its largest high-wavenumber energy fraction was
$0.217\%$, safely below the predeclared $1\%$ pile-up limit.

The always-active dissipative correction was slightly less accurate than the
pure structural model. The sensor remained zero for the entire run, so the
sensed and structural trajectories were identical. The evidence therefore
supports the structural closure; it does not yet support either dissipative
variant.

The first automated result correctly retained a `FAIL` label because its
energy-budget diagnostic exceeded the strict $0.2\%$ tolerance. That output
had been sampled every $1.0$ time unit even though the random force changes on
a $0.2$ time scale, so the power integral was not numerically resolved. This
was tested rather than assumed. The archived interval from $t=50$ to $60$ was
replayed with diagnostics every time step, $\Delta t=0.02$, and was required to
recover the independently archived final state exactly. All fields did so with
zero maximum difference. The correct energy balance,

$$
E(t)-E(t_0)
=\int_{t_0}^{t}\left(P_f-2\nu Z+P_{SGS}\right)\,dt,
$$

then closed within $0.0091\%$ for every model. The original failed diagnostic
and the corrective audit are both preserved. Taken together, this is a pass of
the one-seed stationary **screen**, not publication-level qualification.

[Stationary histories and reference overlays](figures/vpm_les/stage_4b3_stationary_pair_histories.png) ·
[time-mean spectrum overlay](figures/vpm_les/stage_4b3_stationary_pair_spectra.png) ·
[model-error comparison](figures/vpm_les/stage_4b3_stationary_pair_errors.png) ·
[time-step energy-budget audit](figures/vpm_les/stage_4b3_budget_recheck.png)

### Existing VPM reference calculation

Before adding the new closure, the current particle solver was checked using
its complete raw vortex-ring trajectory. This calculation is independent of
the spectral LES research code. The transposed-stretching VPM result has a
$3.84\%$ speed error against Saffman's analytical viscous-ring law, while ring
radius, linear impulse, and tube circulation drift by $0.155\%$, $0.048\%$,
and $2.74\%$. The raw HDF5 sequence contains 24 snapshots through 600 steps.

This supports the health of the base VPM for this smooth unbounded flow. It
does not support the structural model: the curve called “LES” in the existing
tutorial uses the older Smagorinsky viscosity.

[VPM ring speed against Saffman theory](../tutorials/VPM/vortexRing/figures/vortex_ring_motion.png) ·
[VPM ring conservation](../tutorials/VPM/vortexRing/figures/vortex_ring_circulation.png)

### First particle-coupling gate

An explicit LES torque must change particle circulation according to

$$
\frac{d\boldsymbol\Gamma_p}{dt}
=V_p\boldsymbol g_{SGS}(\boldsymbol x_p).
$$

This mapping was tested with a manufactured divergence-free torque whose total
circulation source is zero and whose exact linear-impulse source is

$$
\frac{d\boldsymbol I}{dt}
=\left(0,0,\frac{27}{512}\right).
$$

The production M4′ remeshing weights were used at four particle resolutions.
For a smooth, exactly volume-preserving shear of the particle lattice, the
field error converged at order $3.02$ and reached $0.033\%$ at $32^3$ in both
single and double precision. M4′ preserved the applied circulation and impulse
to numerical precision, and the recovered impulse source differed from theory
by $0.0007\%$.

An intentionally harsher test independently displaced particles while leaving
their volumes unchanged. It failed: $15\%h$ random displacement produced
$12.0\%$ local torque error without convergence, although global invariants
were still preserved. This failed result is retained. It means the production
coupling must apply the structural torque on the GBD remeshing grid and must
monitor particle disorder; it must not simply evaluate a torque on an
arbitrarily disordered equal-volume cloud.

[Manufactured VPM coupling and theoretical impulse](figures/vpm_les/stage_5a_vpm_torque_coupling.png) ·
[retained random-disorder failure](figures/vpm_les/stage_5a_vpm_torque_coupling_random_fail.png)

## Reproducible research materials

- Formulation and frozen-field tests:
  `scripts/experiments/stage_4a_formulation.py` and
  `scripts/experiments/stage_4a_results.json`.
- Exact-solution verification:
  `scripts/experiments/stage_4b0_exact_verification.py` and
  `scripts/experiments/stage_4b0_exact_results.json`.
- First time-dependent turbulence pilot:
  `scripts/experiments/stage_4b_spectral_pilot.py` and
  `scripts/experiments/stage_4b_pilot_results.json`.
- Nested-grid forcing verification:
  `scripts/experiments/stage_4b1_forcing_verification.py` and
  `scripts/experiments/stage_4b1_forcing_results.json`.
- Primary stationary-forcing source:
  `docs/eswaran_pope_1988_forcing.pdf`.
- Resolved-reference forced-turbulence pilot:
  `scripts/experiments/stage_4b1_forced_hit_pilot.py` and
  `scripts/experiments/stage_4b1_forced_hit_64_32_results.json`.
- Stationary-reference qualification:
  `scripts/experiments/stage_4b2_stationary_reference.py` and
  `scripts/experiments/stage_4b2_stationary_reference_results.json`. The
  shorter failed attempt is retained as
  `scripts/experiments/stage_4b2_stationary_reference_short_results.json`.
- Checkpointed stationary model screen:
  `scripts/experiments/stage_4b3_stationary_pair.py` and
  `scripts/experiments/stage_4b3_stationary_pair_results.json`.
- Dense budget follow-up:
  `scripts/experiments/stage_4b3_budget_recheck.py` and
  `scripts/experiments/stage_4b3_budget_recheck_results.json`.
- Manufactured VPM torque-coupling gate:
  `scripts/experiments/stage_5a_vpm_torque_coupling.py` and
  `scripts/experiments/stage_5a_vpm_torque_coupling_results.json`. The original
  independent-random-jitter failure is retained in
  `scripts/experiments/stage_5a_vpm_torque_coupling_random_fail_results.json`.
- Restartable raw states: `artifacts/vpm_les/stage_4b3_seed20260817`. The local
  archive contains 13 checkpoints (124 MB). All 26 state/metadata files pass
  their SHA-256 checksums, and a load-and-continue test reproduces every field
  bit for bit. This directory is ignored by Git and is not yet a redundant
  off-machine backup.

## Progress and work remaining

Updated: 2026-08-16. **Current task:** implement and verify the nonperiodic
fixed filter on the GBD remeshing grid, then run the structural torque in the
raw-backup vortex-ring VPM before returning to the larger spectral campaign.

- [x] Recover the exact filtered-vorticity equation from primary literature.
- [x] Reject coefficient models that cannot represent the exact torque.
- [x] Select and freeze the DIAD structural candidate.
- [x] Gate A: verify mathematical and discrete identities.
- [x] Gate B.0: reproduce exact time-dependent solutions.
- [x] Gate B pilot: complete the reduced Taylor–Green calculation.
- [ ] **Gate B.1: spectral turbulence validation — active.**
  - [x] Verify identical nested-grid forcing, isotropy, temporal correlation,
    and the theoretical forced energy balance.
  - [x] Complete a low-Reynolds-number transient pilot with a reference that
    independently passes its resolution check.
  - [x] Qualify a statistically stationary reference by checking energy drift,
    power input against dissipation, isotropy, and two small-scale resolution
    measures over the final ten turnover times.
  - [x] Run a stationary $64^3/32^3$ paired screen using the qualified
    five-turnover development and ten-turnover measurement protocol.
  - [x] Save restartable raw fields and verify exact restart/continuation.
  - [x] Diagnose the initially under-sampled energy budget and repeat it at
    every time step without changing the archived trajectory.
  - [ ] Repeat the stationary screen for two independent random seeds. Retain
    structural DIAD as the primary candidate; do not claim evidence for the
    inactive sensor.
  - [ ] Test the fixed truncated reconstruction in single and double precision
    because its largest condition number in the stationary screen was
    $6.89\times10^{11}$.
  - [ ] Forced homogeneous turbulence at two Reynolds numbers, meaning two
    ratios of inertial to viscous effects, and three random initializations.
    Run each for at least ten large-eddy turnover times—the characteristic time
    of the largest energetic motion.
  - [ ] Taylor–Green flow through maximum dissipation and two turnover times
    beyond it.
  - [ ] Decaying homogeneous turbulence initialized independently of the AGARD
    field already used during development.
  - [ ] First resolution pair: $128^3$ reference and $64^3$ LES.
  - [ ] Repeat the decisive cases at $256^3/128^3$.
  - [ ] Compare against no model, fifth-order approximate deconvolution,
    continuously active eddy viscosity, structural and sensed DIAD, and a
    recognized functional LES baseline.
- [ ] Gate C: test the unchanged model on unseen types of turbulence.
- [ ] Gate D: verify that GBD represents molecular diffusion correctly without
  receiving any LES viscosity.
- [ ] **Gate E: couple the structural torque to VPM — active.**
  - [x] Verify the circulation-source equation, M4′ transfer, theoretical
    impulse, precision, refinement, and particle-deformation sensitivity.
  - [ ] Replace the periodic FFT filter by a padded, nonperiodic Gaussian
    filter on the GBD grid. Require agreement with the periodic formulation in
    a large interior region and invariance when padding is increased.
  - [ ] Reproduce a manufactured complete structural torque—not merely a
    prescribed source—after particle scatter, filtering, deconvolution,
    derivatives, circulation update, and regeneration.
  - [ ] Run the vortex ring with no SGS and structural DIAD using molecular-only
    GBD. The closure must remain negligible as resolution increases, must not
    worsen the Saffman speed or conservation errors by more than $2\%$
    absolute, and must save restartable raw particles.
  - [ ] Run a genuinely turbulent VPM flow against an external DNS or
    experimental reference. This is required before any journal claim about
    LES performance.
- [ ] Gate F: run full vortex-particle simulations against reference data.
- [ ] Gate G: measure cost and precision sensitivity, then archive every input,
  configuration, seed, result, and figure needed for a journal paper.

## Why the remaining gates are ordered this way

### Gate B.1 — Does the model work when turbulence evolves?

**Motivation.** A model can correlate well with a frozen field and still become
unstable or systematically remove the wrong amount of energy over time.

**Test.** Compare each LES directly with a filtered direct numerical simulation
(DNS), meaning a more highly resolved calculation used as the reference. Plot
energy, enstrophy, dissipation, energy spectra, scale-by-scale transfer,
backscatter, sensor activation, and reconstruction conditioning over time.
Theoretical energy balances will be overlaid wherever an exact balance exists.

**Pass.** No instability or high-wavenumber pile-up; mean energy, dissipation,
and resolved spectra within $10\%$ of filtered DNS; correct forced or decaying
energy balance; and no coefficient changes between cases. The sensor is kept
only if it improves time-integrated spectral error over the structural model in
at least $80\%$ of paired runs.

### Gate C — Does it generalize beyond the flows used to develop it?

**Motivation.** Good performance on familiar homogeneous turbulence may be
overfitting rather than general physics.

**Test.** Lock the model and evaluate unseen homogeneous shear, another Reynolds
number, and one wall-bounded or free-shear dataset. Use independent time blocks
and report $95\%$ confidence intervals.

**Pass.** At least $80\%$ of cases have torque correlation above $0.75$,
transfer ratio between $0.8$ and $1.2$, and lower shell-transfer error than the
simple deconvolution and continuously dissipative alternatives. A failed flow
class narrows the paper's claim; it does not permit retuning.

### Gate D — Is molecular diffusion still correct?

**Motivation.** LES and molecular viscosity represent different physics. Their
numerical implementations must remain separate.

**Test.** Use Gaussian-vortex and vortex-ring diffusion, three grid spacings,
three time steps, and two particle core sizes. GBD receives molecular $\nu$
only. Check analytical diffusion rate, circulation, linear impulse, splitting
error, and particle-pruning loss.

**Pass.** Expected convergence with refinement, conservation errors below the
declared numerical tolerance, and less than $5\%$ LES change when core size is
varied at fixed resolved field, with that change decreasing under refinement.

### Gate E — Can the spectral model be transferred to particles?

**Motivation.** The equations may be correct while interpolation, particle
irregularity, boundaries, or finite precision corrupt their evaluation in
OpenONDA.

**Test.** Convert frozen turbulence fields to particles, evaluate the closure on
the fixed GBD lattice, and compare the resulting torque with the spectral
implementation. Refine $h$ and vary precision, particle jitter, pruning, and
boundary padding.

**Pass.** At the finest grid, torque correlation at least $0.95$, relative
$L_2$ error at most $0.10$, and monotonic improvement with refinement. Failure
here stops particle integration.

### Gate F — Does the complete vortex-particle LES predict a real flow?

**Motivation.** Stability alone is insufficient; the particle solver must show
converged physical statistics.

**Test.** Add a research-only DIAD path that applies modeled torque directly to
particle circulation. Run spatial and time-step refinements for a mixing layer
or turbulent wake and compare with high-quality numerical or experimental
reference data.

**Pass.** Stable budgets, controlled particle count, converged results,
statistics within $10$--$15\%$ of the reference, and agreement with the
spectral implementation within its uncertainty.

### Gate G — Is the work publishable and reproducible?

Measure runtime, memory, conditioning failures, activation statistics, boundary
sensitivity, and precision sensitivity. Archive raw inputs, random seeds,
configurations, the exact code version, and one command that regenerates every
table and figure.

## Stop conditions

- Remove the sensor if it creates persistent bias or is consistently worse
  than the structural model.
- Do not integrate the closure into production OpenONDA if the particle/grid
  bridge does not converge.
- Return to the model equations, rather than tuning individual cases, if the
  fixed reconstruction solve is unstable or precision dependent.
- Do not claim a validated LES from stability alone; the energy and
  scale-by-scale transfer must also agree with independent references.

Production solver code remains unchanged until the spectral and
spectral-to-particle gates pass.
