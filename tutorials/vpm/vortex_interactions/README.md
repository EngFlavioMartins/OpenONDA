# Vortex-ring interactions

This tutorial defines two deliberately severe vortex-ring interactions. In the
leapfrogging case, two equal-signed coaxial rings alternately contract, expand,
and pass through one another. In the collision case, two opposite-signed rings
travel towards one another and form strongly distorted vortex tubes. Both
flows amplify azimuthal disturbances and test whether the particle field
remains usable once a classical fixed-core VPM and scalar LES viscosity cease
to represent the vortex-tube dynamics.

The reported comparison isolates the governing formulation: classical LES,
reformulated VPM, and reformulated VPM with an anisotropic vortex-stretching
subfilter model. Earlier splitting, Pedrizzetti realignment, and sparse
remeshing trials remain under `runs/calibration/`; they are useful failure
evidence but are no longer main-figure methods. Historical DNS outputs are
also retained and excluded from the reported figures.

## Physical and numerical setup

Each ring uses the single-ring discretization from `vortex_ring/`:

| Quantity | Value |
|---|---:|
| Ring radius, `R0` | `1.0` m |
| Initial core radius, `a0` | `0.1` m |
| Tube circulation, `Gamma0` | `pi` m2/s |
| Circulation Reynolds number, `Gamma0/nu` | `3000` |
| Particle spacing, `h` | `0.035` m |
| Particle core radius, `sigma_p` | `0.07` m |
| Particles per ring | `8772` |

The two-ring calculations therefore start with `17544` particles. This is the
same concentration, overlap, kernel, and physical viscosity used by the
single-ring calibration. All new calculations use transposed stretching,
coupled SSP-RK3 integration, Core Spreading, and a Barnes--Hut treecode with
opening angle `0.30`. The classical LES uses `C_s = 0.20`; the anisotropic SFS
case uses a calibrated no-backscatter coefficient `C_d = 0.001` without an
additional scalar eddy viscosity. The initial centreline disturbance contains
modes 1--24 with a total root-sum-square amplitude of `0.025 R0` and fixed
per-ring random seeds.

The nominal time step is

```text
Delta t = 20 h^2 / Gamma0
```

or `Delta t Gamma0 / R0^2 = 0.0245`. This is smaller than the `0.02 s` step of
the calibrated single-ring case, and the coupled solver subcycles further when
the strain or displacement bound requires it. The common request is 1200
steps; flow-integral and ring diagnostics are sampled every five steps. The
solver stops if the peak particle-strength magnitude exceeds 50 times its
initial value. That guard marks numerical loss of resolution; it is not a
physical transition criterion.

## Kinematic reference

The leapfrogging trajectory plot includes the LBM core-centre paths of Cheng,
Lou, and Lim (2015) at `Re_Gamma = 3000`, `a0/R0 = 0.1`, and `h0/R0 = 1`.
Those values match the present Reynolds number, core size, and initial spacing.
The comparison is kinematic rather than pointwise because the literature case
and the present case do not use the same azimuthal perturbation spectrum. The
digitized vector data and their provenance are stored in
`assets/references/`.

The first three LBM overtakes occur at midpoint positions `x/R0 = 1.292`,
`3.596`, and `4.898`. Rapid additional crossings begin near `x/R0 = 5.4` as
the coherent two-ring description breaks down, and the reference extends to
`x/R0 = 7.3`. `assets/check_leapfrogging.py` reports these pass locations,
radius-path RMSE, and whether a candidate reaches the breakdown region.

## Running

`allrun.sh` is the single hard-coded campaign driver. It runs the reported
cases sequentially and then calls the plotting script:

| Interaction | Method | Case directory |
|---|---|---|
| Leapfrogging | LES | `leapfrog_les` |
| Leapfrogging | Reformulated VPM | `leapfrog_les_rvpm` |
| Leapfrogging | Reformulated VPM + SFS | `leapfrog_les_rvpm_sfs` |
| Collision | LES | `collide_les` |
| Collision | Reformulated VPM + SFS | `collide_les_rvpm_sfs` |

```sh
./allrun.sh
```

Use `allclean.sh CASE_NAME` to remove one run, or `allclean.sh --all` to remove
all generated results. Existing results are never overwritten silently.

## Diagnosis and model selection

The long calibration campaign showed that the original failure was not only a
late particle-quality problem. Classical VPM holds each core radius fixed
during stretching. That violates the local mass and angular-momentum
relations of a material vortex tube, so splitting, realignment, or remeshing
can postpone the resulting growth without correcting its source. This is the
same distinction derived in the [reformulated VPM
work](https://arxiv.org/abs/2206.03658).

For the selected reformulation, resolved stretching advances strength and
core size together:

```text
d sigma/dt = -(1/5) sigma (S . Gamma_hat) / |Gamma|
d Gamma/dt = S - (3/5) (S . Gamma_hat) Gamma_hat
```

Here `S` is the configured discrete vortex-stretching rate. Coupled RK3 uses
the stage core radius in both the velocity and stretching operators. A
qualification run without the anisotropic SFS term reached its first pass at
`x/R0 = 1.328`, compared with `1.292` in the LBM reference; its two radius-path
RMSE values through step 205 were `0.0160 R0` and `0.0081 R0`. This correct
first-pass kinematics is the direct evidence that the strength/core coupling,
rather than stronger late damping, repairs the leading trajectory error.

Reformulation alone does not model unresolved stretching. The final LES term
therefore follows the anisotropic operator in the official [FLOWVPM
implementation](https://github.com/byuflowlab/FLOWVPM.jl): velocity-gradient
differences within four source-core radii estimate the missing stretching and
only forward transfer is retained. The compact sum uses the existing LBVH, so
its cost scales with local neighbours rather than all particle pairs.

The coefficient is deliberately small. `C_d = 1.0` collapsed the initial
energy, and `C_d = 0.01` remained too dissipative. `C_d = 0.001` retained the
first-pass trajectory and early energy budget; its magnitude is also the net
attenuation approached by the three-level controlled model when its published
scale factor is near `2/3`. This is a calibrated constant approximation, not a
claim that the local dynamic coefficient is uniform.

Pedrizzetti relaxation removes the remaining discrete strength--vorticity
misalignment with `f = 0.05` every five steps. Each particle first keeps its
strength magnitude. A minimum-norm correction then restores total vector
strength, linear impulse, and Gaussian-core angular impulse exactly. The
correction was `0.10--0.15` percent of the field over the 30-step qualifier;
energy decreased by `0.88` percent and vector-strength closure stayed below
`4e-10` per event. The uncorrected version accumulated a net strength of
`0.07 Gamma0` in only 30 steps. No scalar Smagorinsky viscosity is added to
this case.

### Production outcome

The retained leapfrogging calculation remained numerically bounded through
the breakdown region at `x/R0 = 7.09` with the original 17544 particles. Its
three material-ring crossings were `x/R0 = 1.283`, `3.823`, and `6.932`,
compared with the LBM values `1.292`, `3.596`, and `4.898`. The first two are
within the `0.5 R0` acceptance band; the third is not. Kinetic energy peaked
only `2.30` percent above its initial value and ended `30.5` percent lower,
while net vector-strength drift stayed below `5e-7 Gamma0`. The calculation
therefore demonstrates stable breakdown capture, but not a quantitatively
validated third leapfrog.

The delayed third crossing is the remaining model limitation. Reformulation
removes the fixed-core inconsistency and the anisotropic SFS term prevents
unresolved stretching from running away. Repeated direction relaxation then
keeps the particle field usable, but after the second pass it also lengthens
the leapfrog cycle. By `x/R0 = 7.09`, the divergence error had risen to `0.213`
and the strength-magnitude sum by `19.8` percent even though global moments
were closed. This is a conservative but phase-inaccurate solution: stronger
relaxation improves survival at the cost of material-ring kinematics, while
weaker treatments in the calibration archive lose resolution earlier. None of
the tested fixed-particle settings satisfies both requirements simultaneously.

The matching collision calculation remained symmetric through annihilation.
Its energy fell from `23.30` to a minimum of `1.36` at step 980. It was stopped
at step 1140 after energy rebounded to `2.04`, the strength-magnitude sum had
grown by `76.6` percent, and the divergence error reached `0.400`. Those late
trends belong to the diffuse post-collision particle field, not to a credible
physical recovery of the rings. Results through the energy minimum are the
representative collision solution.

The numerical audit also found that the interaction driver still used RK2
although the single-ring calibration used RK3. All new cases now use coupled
SSP-RK3. The particle concentration is already identical to `vortex_ring/`,
and the nominal interaction time step is smaller; neither is changed unless
the LBM comparison demonstrates a resolution error.

### Rejected symptom treatments

Particle splitting is checked every 25 steps and refines particles whose
strength exceeds three times their lineage reference. It reaches step 495
with 19382 particles, a 10.5 percent increase over the initial cloud. A factor
of five adds only 1.6 percent particles but ends at step 335. The factor-three
case is retained because it gives the longest credible local-method survival
without splitting every time step or causing runaway particle growth.

Realignment applies a Pedrizzetti relaxation factor of 0.005 every 25 steps.
It reaches step 385 with the original 17544 particles, the longest credible
leapfrogging result at fixed particle count. Factors of 0.10 and 0.02 inject
energy and cause substantial impulse drift; 0.01 is cleaner but ends five
steps earlier. The retained weak relaxation reduces late strength-vector
misalignment without behaving as an artificial energy source.

Sparse remeshing is useful only while the cloud remains moderately distorted.
The first reset, on a grid with spacing `0.040 R0`, changes kinetic energy by
about -0.45 percent and maps 17544 particles to 24116. A second capacity reset
with spacing `0.045 R0` changes energy by about -0.36 percent and produces
33267 particles. A third reset reaches the 60000-particle ceiling and removes
about 8 percent of energy and 3 percent of enstrophy; it is rejected. The
reported remeshing-only history therefore ends before that third event.

The best classical-VPM combination used each operation only in the regime it
addresses:

- weak Pedrizzetti realignment acts every 25 steps;
- factor-three splitting resolves stretched filaments through step 550;
- at most two `0.055 R0` remeshes reset the cloud after the divergence error
  exceeds 0.22, with 30000- and 45000-particle ceilings;
- after step 550, a local residual viscosity opposes positive vortex stretching.

Each remesh must preserve enstrophy to the float32 reduction tolerance, close
the net vector strength, and remove no more than 5 percent of kinetic energy.
The late residual viscosity is

```text
nu_s = C Delta^2 max(alpha . S alpha / |alpha|^2, 0)
```

with `C = 1.6` initially. Every five steps an energy-budget feedback adjusts
`C` by at most 25 percent and caps it at 8. This damps only positive local
stretching production; it does not apply a uniform late-time diffusion.

| Case | Last step | Final particles | Interpretation |
|---|---:|---:|---|
| Leapfrogging LES | 305 | 17544 | Baseline loss of filament resolution |
| Leapfrogging LES + splitting | 495 | 19382 | Best local refinement trial |
| Leapfrogging LES + realignment | 385 | 17544 | Best fixed-cloud trial |
| Leapfrogging LES + remeshing | 495 | 33267 | Two accepted sparse resets |
| Leapfrogging LES + combined | 680 | 45000 | Best staged calibration before renewed growth |
| Collision LES | 660 | 17544 | Baseline collision survival |
| Leapfrogging rVPM + SFS | 820 | 17544 | Three crossings and breakdown; third pass delayed |
| Collision rVPM + SFS | 1140 | 17544 | Stable annihilation; stopped at late energy rebound |

None of these classical-VPM variants reached the three coherent LBM passes.
They are retained because they quantify the cost and numerical contamination
of treating the symptoms: splitting and remeshing increase particle count,
realignment can inject energy, and repeated remeshing alters material-ring
circulation. They are not reported as successful stabilization methods.

Scalar LES dissipation cannot add Lagrangian degrees of freedom, repair a
strength direction that has drifted away from the local vorticity, or restore
a cloud whose overlap has collapsed. The three interventions address those
symptoms, but none restores the missing local vortex-tube relation. This is why
their survival improves without converging to the LBM leapfrogging sequence.

Representative selected results are kept in `solution/` and `samples/`;
parameter trials are retained under `runs/calibration/`. The principal
leapfrogging backups are `leapfrog_factor3_interval25_no_projection/`,
`leapfrog_realign_factor005_interval25/`,
`leapfrog_remesh_spacing040_capacity045_second_event_accepted/`, and the
`leapfrog_combined_*` staged-calibration directories.

## Figures

Generate PNG figures by default, or request PDF explicitly:

```sh
./allplot.sh
./allplot.sh pdf
```

The figure families are:

- `leapfrogging_trajectory` and `collision_trajectory`: strength-weighted
  material-group centroid and covariance radius;
- `leapfrogging_energy` and `collision_energy`: reconstructed kinetic energy;
- `leapfrogging_circulation` and `collision_circulation`: the circular-tube
  circulation estimate, normalized by the relaxed step-15 sample;
- `leapfrogging_stability` and `collision_stability`: peak particle-strength
  magnitude and the 50-times solver guard.
- `leapfrogging_conservation` and `collision_conservation`: particle count and
  drift of vector strength, linear impulse, and Gaussian angular impulse;
- `leapfrogging_resolution` and `collision_resolution`: overlap, divergence
  error, and strength--vorticity misalignment.

The trajectory estimator remains meaningful while each initial material group
tracks one coherent tube. After reconnection or complete mixing, group IDs are
material labels rather than unique vortex-core identifiers; the late curve
must then be interpreted together with field visualizations and the other
diagnostics.
