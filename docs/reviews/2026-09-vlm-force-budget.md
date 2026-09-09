# Flat-plate force budget: unsteady pressure and resolved loading

9 September 2026. **Audit in progress.** The native v19 bound/free exchange has
passed its strength checks. Existing v19 samples report quasi-steady
Kutta–Joukowski forces. A native unsteady-pressure implementation has passed analytical, restart and
coupled checks, including full static8°/static5°/moving5° tutorial runs. The current
flat reference uses direct wake induction after the separate free/free tree
approximation diagnosis in the circulation budget. Direct static12° has passed
its complete force, strength and impulse checks; the rest of the direct-reference
sweep is running sequentially. Later tutorial cases remain stopped.

## Evidence from native data

Three bounded refinements ran sequentially from the actual flat-plate tutorial
directory, with native outputs in `solution/qualification/exchange_v19/` and
`samples/qualification/exchange_v19/`. They use the authored case physics and the
frozen v19 installation. The original completed coarse run supplies its existing
step30/t0.375 sample; it was not repeated.

| Case at t=0.375 s | dt [s] | Span panels per half | CL | CD | Peak vector closure |
|---|---:|---:|---:|---:|---:|
| Coarse | 0.0125 | 14 | 0.642301934 | 0.017604000 | 4.8001e-6 |
| Time half | 0.00625 | 14 | 0.636719856 | 0.017593348 | 1.8529e-6 |
| Time quarter | 0.003125 | 14 | 0.634918676 | 0.017575954 | 1.8061e-5 |
| Span half | 0.00625 | 28 | 0.631315321 | 0.017444308 | 3.6687e-5 |

All cases have eight chordwise panels and pass the unchanged1e-4 strength gate.
Successive time refinements reduce the lift difference from0.869% to0.283%.
Drag differences are small but do not establish a formal convergence order.
These are developing wakes; they cannot be compared with steady theory as
though they had reached a plateau.

Over the common interval t=0.0625–0.375 s, differentiating the native coupled
linear impulse gives roughly6–7% more lift than integrating the native force
history. The force evaluator currently uses only instantaneous Kutta–Joukowski
loads at the lifting bound legs. It has no time-dependent pressure contribution.

## Independent pressure-time check

For this stationary planar plate, each horseshoe increment Gamma_i represents
a potential jump on the on-wing area downstream of its bound leg. From the
native strip widths and bound-point positions, the relevant integral is

```
M(t) = rho sum_i |Gamma_i(t)| width_i (x_TE - x_bound,i).
```

The absolute value here accounts for the reversed orientation of the mirrored
half of this symmetric positive-incidence test. It is **not** a valid general
solver formula for arbitrary signed circulation or motion. The native
implementation must use oriented area vectors and signed circulation.

The integrated normal pressure-time contribution over [t0,t1] is M(t1)-M(t0).
This requires no differentiation of noisy force samples and no checkpoint
extraction. Adding this independent pressure contribution to the time integral
of native KJ forces substantially closes the impulse budget:

| Case | Mean added normal force [N] | Impulse / corrected lift | Impulse / corrected drag |
|---|---:|---:|---:|
| Coarse | 16.51286 | 1.007605 | 0.983843 |
| Time half | 18.57675 | 1.003941 | 1.010892 |
| Time quarter | 19.09539 | 1.002536 | 1.015635 |
| Span half | 18.24451 | 1.003874 | 1.021027 |

This identifies a force-model omission; it is not evidence of remaining large
vorticity loss. Neither the simulation forces nor the fluid evolution were
modified by this audit. The native-source comparison figure is saved under
`tutorials/vpm/flat_plate/figures/qualification/flat_unsteady_force_audit`.

The unsteady Bernoulli term involving the time derivative of the surface
potential jump appears explicitly in [Roccia et al., Wind Energy Science9
(2024), Appendix C3](https://wes.copernicus.org/articles/9/385/2024/wes-9-385-2024.html).
Its vortex-ring strength must be mapped correctly to this solver's chordwise
horseshoe increments. Multiplying every local horseshoe increment derivative by
its physical panel area would not perform that mapping.

Related implementations separate quasi-steady KJ and unsteady-circulation
forces. FLOWUnsteady documents a separately stored unsteady contribution and
allows it to be omitted from reported total forces; that deliberate reporting
choice does not establish that the term is physically zero. See its [force
calculator documentation](https://flow.byu.edu/FLOWUnsteady/api/flowunsteady-monitor/).

## Native implementation requirements

The implementation follows the audited mapping below. Analytical tests compare
its physical-panel partition against independently integrated whole downstream
pressure patches. Keep the public tutorial setup simple and put the calculation
in the solver.

1. The physical panel's potential jump is the cumulative upstream circulation,
   with an additional local Gamma jump at its quarter-chord bound leg. Split
   each physical panel there: the fore portion uses
   `G_before = G_cumulative - Gamma`, and the aft portion uses `G_cumulative`.
   Use current minus previous accepted values divided by the physical step.
   This distributes pressure on the actual panels, preserving meaningful
   chordwise pressure/loading samples. Do not assign the entire downstream
   horseshoe-area force to its upstream physical panel.
2. Integrate each fore/aft quadrilateral as oriented triangles. For the right
   half of the present plate, the vertex orders
   `[LE_left, bound_left, bound_right, LE_right]` and
   `[bound_left, TE_left, TE_right, bound_right]` give positive normal area.
   Mirroring reverses both the signed circulation and oriented area. Validate
   negative incidence, reflected geometry and arbitrary rigid rotations.
3. Integrate the moment at each triangle centroid. Existing KJ forces act at
   bound midpoints. Store the additional moment relative to that point and
   include it in both total/reference/quarter-chord moments and per-surface
   pivot torque/power. Applying the pressure force at the bound midpoint would
   give a wrong pitching moment even if total force matched.
4. Preserve the explicitly available quasi-steady force model. Add a clear
   native unsteady option and exercise it in the flat-plate setup. Do not add
   a tutorial-side force reconstruction. Update force-method documentation
   and native output semantics; do not overclaim complete added-mass accuracy
   without an independent moving/accelerating-plate test.
5. Persistent moment fields require exact restart handling and a numerical
   format/version change if the restart schema changes. Existing v19 results
   remain preserved. Force changes should leave particle transport identical
   unless an explicitly coupled force-dependent model is selected.
6. Force CSV cadence is already independent of `VLMSampler` VTK cadence:
   `VLMSetup.logging_interval_steps` controls native force/distribution output.
   For startup/motion validation, use force output every step while keeping
   geometry samplers and numerical backups less frequent. No new tutorial
   sampling implementation is needed.

Meaningful verification: analytical constant/ramping potential jump on a flat
plate; chordwise partition telescoping; force and moment covariance under rigid
motion; mirrored/negative-incidence signs; independence from origin; exact
restart; density scaling; finite no-op for constant circulation; and a real
coupled impulse comparison with the new native force output. Existing force,
loading, per-surface power and output-contract tests must remain valid.

## Remaining steady-force question

The fixed inviscid lattice diagnostic also compared near-field KJ drag with an
odd-Fourier-series estimate from its resolved spanwise circulation. Their gap
decreases from5.14% to2.92% to1.68% as span resolution goes14→28→56 panels per
half; doubling chord resolution has little effect. Native drag itself changes
less than0.06%. This supports a spanwise discretization contribution, not an
arbitrary force rescaling or a demand for exact lifting-line agreement.

On-wing trailing-leg force terms should be assessed separately if material.
Some lifting-surface derivations include all three on-wing horseshoe legs;
actuator-line implementations may deliberately omit the trailing legs. Do not
change that modeling choice solely because one reference curve then fits.

The audit workspace retains the raw native-data readers, all numerical reports,
conditioning/truncation evidence and reproduction scripts. See the
[working checklist](2026-09-vlm-todos.md) for current execution state.

## Candidate v20 native results

The analytical/restart/solver batch passed50 checks and the output/Galilean batch
passed37 checks (the latter includes the extended force/moment frame test).
The bounded native60-step run completed at t.375. Its integrated pressure-time
load matches the independently integrated sampled potential jump to2.49e-7 N s
(out of5.805235 N s). Over common t.0625–.375, surface/fluid impulse discrepancies
are -0.3285% lift and -1.0018% drag using interval integration of the backward
pressure derivative and trapezoidal integration of KJ loads. Native wake changes
from v19 are at f32 reduction levels. Details are in the working checklist.

The full native static8° run completed192 steps/t2.4. CL=.68400633 and
CD=.01531379; last5-chord CL variation is0.1163%, below0.2%. Full-vector
strength closure is3.9337e-5, below1e-4. Late-time fluid-impulse lift agrees with
reported force within0.066%; drag differs1.61%. Across the complete sampled
window, integrated total lift/drag differ -0.134%/+0.639% from fluid impulse.
These are measured finite-discretization residuals, not exact force identity claims.
The native impulse plot is now part of the tutorial's normal allplot launcher.

### Read-only on-wing force quadrature

A probe restored the final native checkpoint without advancing a step and
evaluated the existing complete velocity field on the two on-wing trailing legs.
Eight/sixteen Gauss points per leg give added vertical force.125663/.125662 N,
about0.037% of the340 N vertical load. Their contribution to wind-axis drag is
about0.23% of total drag. They do not explain a large missing load in this plate
case; there is no basis here to replace the selected bound-leg force model.

The same diagnostic's bound-leg quadrature is **not a converged force reference**:
Fx changes -38.94→-38.35→-37.74 N with4/8/16 points, whereas the native midpoint
model gives -40.02 N. The nearly singular trailing filaments at strip edges
make integrating the unrefined, discontinuous strip representation a different
operation from refining the VLM's force collocation. Do not mistake this diagnostic
for an improvement or change force output to follow it. The existing physical
mesh-refinement and near-/far-wake comparisons are the relevant discretization
evidence. Keep the raw quadrature values as a limitation of this probe.

No delta, turbine or quad run has resumed. The rest of the flat sweep and its
static/moving5° comparison still require current native results.

## Direct-induction reference

The force formulation is unchanged when selecting the direct particle backend.
The full static12° comparison isolates that induction accuracy choice: final CL
changes by-.000121%, CD by+.02906%, and CMc4 by-.35244% from the preceding tree
result. The native direct result is CL=1.020440869, CD=.0341096465 and
CMc4=.004168679. Its final5-chord CL range is.11716%, below the unchanged.2%
criterion. The full-vector strength closure is4.91637e-6, below1e-4.

Over the native sampled window t=.0625–2.375 s, interval-integrated surface load
differs from fluid impulse by-.19437% in lift and-.13511% in drag. Over the late
window t=1.875–2.375 s, the differences are-.12173% and+.41726%. Inboard80%
sectional lift/downwash L2 differences from lifting-line are3.061%/5.137%; full
span differences are3.959%/23.771%. Exact pair summation removes tree truncation
from this small reference problem; it does not remove finite-chord, tip, time-step
or force-discretization differences from the analytical comparison.

The ongoing direct sweep's broader native reports are in
`flat-direct-v20-polar-progress.json` in the retained audit workspace. These
reports use proper backward-interval pressure integration, unlike a simple
trapezoidal integration of total force. Whole-window and late-window errors are
reported separately; the static12° numbers above must not be applied to every
angle or frame. See the working checklist for the current completed case set.
