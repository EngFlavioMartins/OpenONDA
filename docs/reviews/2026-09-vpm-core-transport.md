# VPM core transport: physics before stabilization

**The previous stabilization campaign did not establish an LBM match. The main
change in direction is to resolve viscous core deformation, rather than keep
round rings alive.** The new default `allrun.sh` compares molecular-viscosity
controls with corrected initial conditions. None is yet a certified winner.

**The leading candidate is fixed-core GBD with six-point Lagrange remapping.**
It develops elongated cores, entrainment and a strong connection between the
cores, while the matched core-spreading control retains two compact cores.
This holds with bitwise-identical initial particle positions, strengths,
widths, volumes and group IDs (`diagnosis_cs_samecore_fast_h04`).
At t=4.5, the vorticity along the line connecting the two strongest maxima
never falls below 70% of the weaker maximum in the Lagrange run, versus 1.1%
in the core-spreading control. This deformation precedes the Lagrange run's
first particle-cap event at step 303 (t=4.545). Two maxima therefore cannot
be interpreted automatically as two intact rings. This is encouraging evidence
of viscous merging, **not a validated Widnall instability or an LBM match**.
The second large radial excursion remains downstream of the LBM curve; the
transport changes have not yet removed that phase error. No extra viscosity
or distance rescaling was used to make the curves agree.

The larger-capacity Lagrange continuation reached t=6 with **391,790 particles
and zero population-cap events**. It starts from the original run's pre-cap
t=3 snapshot; their velocity fields agree to 0.0063% at t=4.5. The usual
0.01% per-step strength-tail budget remains active, so “no cap” does not mean
no pruning. The continuation took 33.5 minutes of recorded wall time, including
initialization and concurrent work, in addition to its parent's first half.
At t=6, the capped and larger-capacity velocity fields differ by only 0.029%
in relative L2 on 1,024 identical independent targets; the dominant peak is at
x=7.441, r=1.075 in both. Thus the observed merging does not depend on that
population cap. A weak remnant maximum remains, with about 11% of the dominant
peak's vorticity, so this is not a claim that all secondary vorticity vanished.
In the identical-start CS control, the two final peaks remain comparable
(the weaker is 91% of the stronger), and their local core aspect ratios remain
close to one. This CS control completed in 13.3 minutes of recorded wall time.

## Findings that are established

| Check | Measured result | Consequence |
|---|---|---|
| Initial Gaussian, h=0.04, particle sigma=0.08 | Old peak 107.27 versus prescribed 100; retained tail gives about 99.90 | Correct the initial profile before comparing dynamics. |
| Core-spreading diffusion coefficient | Measured variance increase 0.02123376; `4 nu t` is 0.02123334 at t=5.069 | Molecular viscosity is present and has the right coefficient. |
| Periodic versus unbounded initial point velocities | Periodic axial estimate 1.24248; unbounded estimate 1.26446 at nominal core centres | A roughly 1.7% difference warrants a matched-boundary check. These use different discretizations and are not bulk propagation speeds. |
| Translating Gaussian, h=0.04, dt=0.015, 100 steps | M4' peak / exact peak 0.8539; six-point Lagrange 0.9765 | Remapping can create substantial extra damping at this spacing. |
| Same translation test, h=0.02 | Ratios 0.9880 and 1.0021, respectively | Refinement remains necessary; moment conservation alone does not establish field accuracy. |
| Large saved cloud, tree versus direct sums | theta=0.5/order=3: velocity L2 error 0.054%, gradient error 0.130%; induction stage about 12 times faster than theta=0.1/order=1 | A measured cost option makes fixed-core experiments practical. This is a snapshot qualification, not a universal error bound. |
| Direct stretching-rate check on the fixed-core cloud | Relative L2 error 0.217% initially and 0.379% at t=3 for theta=0.5/order=3 | The faster setting also resolves the stretching contraction on the sampled particles. |

The initial profile issue came from discarding 5% of a Gaussian and then
renormalizing its circulation. The tutorial now retains all but 0.01%; the
study records this separately as `--initial-tail`. Existing saved runs retain
their original conditions. The 0.01% initial-tail correction must not be
confused with the per-step GBD pruning budget.
Correcting the tail changes the sampled initial axial velocity by only about
0.35%, so it does not explain the whole trajectory discrepancy. Moreover,
radius versus distance tests the leapfrogging path and its phase; establishing
an absolute propagation-speed error additionally requires position versus time.

## Why core spreading can look inviscid

A growing spherical blob can diffuse correctly while being transported
incorrectly by a strained flow. In the old long control, blob width becomes
most of the reconstructed core width, and the cores remain nearly circular.
The fixed-core controls produce elongated cores and connecting vorticity.
An independent affine-strain calculation also shows that uncorrected core
spreading retains a convection/diffusion error even as its *initial* blob
width tends to zero. This is consistent with the classical inconsistency
identified by [Greengard (1985)](https://www.sciencedirect.com/science/article/abs/pii/0021999185900919).
It does not prove that this is the sole cause of the ring-trajectory mismatch.

The LBM target must also be stated correctly. [Cheng, Lou and Lim (2015)](https://doi.org/10.1063/1.4915890)
use an unperturbed Re=3000 case for the trajectory comparison; loss of repeated
leapfrogging includes viscous deformation and merger. Their separate seeded
example uses an axial mode-eight displacement. Our old seed was radial.
The initializer now supports axial displacement, its tangent-vorticity
component and consistent displaced particle support. General-mode tests also
caught and corrected an azimuthal-origin mismatch between geometry and field
attribution. Axial seeding is available; a matched seeded-breakdown validation
has not yet been completed.

## Code and evidence

- `ViscousConfig.gbd(remeshing_kernel="LAGRANGE6")` selects an experimental
  tensor product of local degree-five Lagrange polynomials. It reproduces
  moments through degree five; it is not Monaghan's M6 kernel, a positivity
  limiter, or rVPM. The default production GBD kernel remains M4'. The
  [high-order remapping literature](https://www.sciencedirect.com/science/article/abs/pii/S0021999111005237)
  motivates moment-preserving remapping; the implemented polynomial and its
  tests define this particular option.
  At t=4.5 it still has small negative undershoots: in the plotted meridional
  window, negative vorticity contributes about 0.10% of the absolute field
  sum (M4': 0.21%). Higher order reduces these errors but does not eliminate
  them or establish convergence.
- Uniform scalar viscosity transfers now avoid an unnecessary nearest-neighbour
  fill. This changes cost, not the transferred value.
- `track_vorticity_cores.py` reconstructs mean azimuthal Gaussian vorticity and
  locates its peaks without material labels. It also records local core aspect
  ratio and exact signed meridional circulation. Peak counting in a finite
  window is a diagnostic, not independent proof of three-dimensional breakdown.
- Tree and stretching controls are explicit in study metadata. The restart
  identity now includes the stretching scheme.

The [trajectory comparison](../../tutorials/vpm/vortex_interactions/figures/study/physics/trajectories.png)
and [final core sections](../../tutorials/vpm/vortex_interactions/figures/study/physics/core_sections_t6.png)
separate field peaks from material-label radius proxies. Labels mix during
regeneration, so their continuing or disappearing curves cannot decide the
winner. Run manifests, raw snapshots and termination reasons are under
`tutorials/vpm/vortex_interactions/study_results/diagnosis_*`; scalar controls,
profile audits and timing qualifications are under `artifacts/vpm_core_transport/`.
The targeted initialization, diffusion, field-reconstruction and tutorial
checks pass, as do Ruff checks on the changed implementation and shell syntax.

The independent axisymmetric finite-difference experiment is a diagnostic,
not an additional reference solution: its late-time oscillations and boundary
approximation have not been sufficiently qualified. The random-walk control
at sigma/h=1 stopped after two steps on the existing divergence limit; that
setting is rejected, not presented as evidence against all random-walk methods.

## Reproduce and decide

From the tutorial directory, `./allrun.sh` now runs unperturbed, no-LES CS,
GBD/M4' and GBD/Lagrange6 controls serially at h=0.04, dt=0.015 to t=6.
All three start from the same particles with sigma/h=1; the wider-blob CS
diagnostic remains a separate saved control.
`--campaign strategies` retains the earlier stabilization experiment.
The current fixed-core controls use sigma/h=1, a marginal overlap setting.
The launcher now reserves 500,000 particles and saves fields every 20 steps;
this is a capacity, not a guarantee that every requested horizon fits.
Inspect cap losses and repeat with smaller h and
adequate overlap before interpreting merger as validated physics. Do not
increase viscosity or fit an axial shift to force agreement.

The acceptance criterion remains: the reconstructed core-radius trajectories
must track the LBM curves through the coherent phase, and deformation/merger
must persist under spacing, timestep, tree-accuracy and pruning checks. Only
then should LES and the correctly directed seed be used to study 3D breakdown
and decide which stabilization is useful.
