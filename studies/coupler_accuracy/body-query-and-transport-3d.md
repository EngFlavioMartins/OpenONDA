# Body-panel field in point queries and particle transport

The selected cube configuration uses panel scope `vpm_boundary_condition`.
Frozen-code inspection and an actual saved-state stage probe establish that
the body field enters VPM target queries and FVM boundary data, while particle
RK stages exclude its velocity and velocity gradient. Adding that field
changes the stage rates measurably. The controlled 20-interval advancing
comparison improves drag-history and velocity errors modestly, and a
verified 40-interval prefix confirms that benefit through physical time `2.5`.
The requested close agreement and developed-wake validation remain unmet.

The [saved-source audit](body_query_transport_gap_3d.py) measures the size of
that field on the independently verified profiles through physical time `4.0`.
Analytical triangle integration and the native double-precision panel kernel
agree to a maximum of `8.327e−17` across all 15 frames. The panel strengths
remain the saved single-precision values.

## Configuration and execution paths

The [matched trial](cube_coupled_trial.py) explicitly constructs the panel
solver with `coupling_scope="vpm_boundary_condition"`. In the frozen
[VPM initialization](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/source/solvers/vpm/core/solver.py:645),
both panel scopes install `_body_induced_fn` for target queries. Only scope
`full` installs the particle-stage body velocity and gradient hooks. The
selected branch sets `physics.body_velocity`, `body_velocity_field` and
`body_velocity_gradient` to `None`.

The [stage provider](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/source/solvers/vpm/physics/stage_rhs.py:624)
adds those hooks when present. This trial has no auxiliary source particles,
VLM or velocity override supplying an equivalent term. The
[panel stepper](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/source/solvers/vpm/coupling/stepper.py:46)
also returns immediately for the selected scope. The FVM boundary update
refreshes the panel solution and explicitly requests body-complete queries.

Consequently, the current RK time-refinement experiment preserves particle
transport driven by particles plus freestream, while its accepted profile and
boundary queries include the panel correction. Its runner's inherited phrase
“including its body contribution” means the existing stage RHS; it must not
be read as claiming an active body term in this selected scope.

## Size of the omitted query contribution

These are vector RMS magnitudes divided by freestream speed, expressed as
percentages. They measure the panel contribution itself, not hybrid/reference
error. Both lines observe the fully 3D flow at `z=0`; off-axis means `y=0.75`.
Small-FVM points are fluid points inside the small box. Near wake means
`1.5 < x ≤ 4`.

| Physical time | Centreline, small FVM | Off-axis, small FVM | Centreline, near wake | Off-axis, near wake |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 1.856% | 0.757% | 0.1090% | 0.0840% |
| 1.5 | 1.831% | 0.730% | 0.1102% | 0.0838% |
| 4.0 | 1.199% | 0.299% | 0.00306% | 0.00244% |

The contribution is measurable near the body. Its small instantaneous value
in the later exterior wake does not account directly for the `2.846% U∞`
centreline wake discrepancy at that time. It also cannot bound the accumulated
effect of different trajectories, stretching and repeated particle replacement.
The audit retains upstream and far-wake regions and compares instantaneous
fields with and without the panel term at the four direct-evaluation times.

## Actual stage probe on the saved particle state

The [qualified stage probe](body_stage_snapshot_reference_3d.py) restores the
canonical VPM backup at physical time `1.5` through the public loader, including
its strict configuration check. All 28,441 particle positions, strengths and
core radii match the accepted profile bitwise. The saved 108 panel strengths
are restored without resolving the body problem. Minimum particle clearance
from the cube is `0.03125`.

Four actual `StageRHS` evaluations compare the original stage, body velocity
alone, body velocity with the native gradient, and the original stage again.
The replay is bitwise identical, and primary particle fields and clocks remain
unchanged. Velocity-only addition leaves circulation rates unchanged; adding
the gradient reproduces the native transposed-gradient contraction within
the stated single-precision accumulation bound.

| Particle region | Particle population | Added body velocity RMS (% U∞) | Native body-gradient relative Frobenius error (%) |
| --- | ---: | ---: | ---: |
| Near body, max absolute coordinate < 1 | 22,501 | 1.36489 | 1.35856 |
| Near wake, 1.5 < x ≤ 4 and max absolute transverse coordinate < 1.5 | 3,127 | 0.19686 | 15.43266 |

Velocity magnitudes use every particle in each region. Gradient errors use
128 deterministically selected particles per region and an independent
double-precision surface-integral reference. Relative error in the body
stretching contribution on those targets is `0.850%` near the body and
`17.804%` in the sampled near wake. These are errors in the body contribution,
not percentages of the complete flow error.

Adding the body velocity reduces the selected near-body stage/query gap
from `0.0122133` to `0.0003830 U∞` RMS. The remaining gap is also present
when comparing stages and queries with the body term excluded. The separate
[particle-stage audit](particle-stage-induction-3d.md) now attributes that
saved-state difference mainly to the hierarchical stage evaluator; arbitrary
point queries use a direct fallback. This does not establish its accumulated
effect on a coupled trajectory.

The [probe record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-stage-reference-at-one-point-five/body-stage-snapshot-reference-3d.json)
retains all stage fields, target arrays, source hashes and checked limits.
No particles or FVM states advance in this probe.

## Independent reference and a qualified analytical gradient

The first probe stopped at its reference-accuracy assertion. Passing `f64`
arrays to the native kernel in a Taichi runtime with default `f32` retains
a rounded literal `4π` denominator. The observed `9.018e−10 U∞` maximum
difference is predicted by the constant ratio `0.9999999721724656`, leaving
only `4.857e−17 U∞` discrepancy. This ratio is computed from the constant,
not fitted to the field. The failed script and artifacts remain intact.

The corrected reference integrates `R/r³` and its full target derivative
`I/r³ − 3 RR/r⁵` over adaptively subdivided triangles. Duffy quadrature at
orders 8 and 12 agrees to `1.325e−15` for velocity and `1.427e−13` for the
gradient. Analytical triangle velocity agrees with quadrature to `3.920e−16`;
refined fourth-order differences agree with the integrated gradient to
`2.383e−11`. The particle runtime remains `f32`; the original gate was not
loosened.

The [analytical study operator](analytical_panel_gradient_3d.py) differentiates
the source kernel's solid-angle and edge-log terms directly in double
precision. It agrees with the independent gradient at all 256 checked
targets to a maximum `1.368e−14`. Maximum antisymmetric entry is `2.054e−15`
and maximum divergence is `8.049e−16`. Rotation, translation, length scaling
and strength scaling checks pass. All 28,441 saved gradients are finite;
the independent surface integral checks cover the selected 256 targets.

The [operator qualification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/analytical-panel-gradient-stage-qualified/analytical-panel-gradient-qualification-3d.json)
does not qualify surface traces, source-edge singularities, or an advancing
trajectory. This remains an experimental CPU operator.

## Advancing comparison

The [body-transport runner](cube_body_transport_3d.py) adds only the native
panel velocity and qualified analytical gradient at temporary RK stage
positions. Panel strengths retain the existing update schedule and remain
fixed within each VPM interval. Panel advancement/shedding stay disabled;
FVM exchange, renewal, RK/GBD time steps and outer stabilization remain at
their original settings. Simply selecting scope `full` would change
additional behavior and would not isolate this contribution.

The disabled-wrapper control is qualified: three-interval histories, six
comparison arrays, 17 FVM entries, 11 boundary-history entries and 11 numeric
VPM datasets match the original control bitwise. Eight independently
reconstructed wall forces agree exactly. The separate schedule verifier
checks six stages, three GBD calls and twelve outer stabilization phases,
with no body callback invoked. Its profile figure has the same hash as the
already inspected split-wrapper control figure.

The enabled 20-interval comparison is now complete and independently checked.
All 20 intervals converge in three sweeps each, with bitwise replay of the
24 recorded first-map state/clock entries in every interval. The same
small FVM box, native `h=0.0625` cells, laminar viscosity and time steps are
retained; the full reference advances independently.

| Measurement, physical time 0.5–1.5 | Original particle transport | With body velocity/stretching |
| --- | ---: | ---: |
| Drag-history RMS relative error (%) | 0.555068 | 0.521061 |
| Maximum absolute relative drag error (%) | 1.481333 | 1.490997 |
| Final hybrid Cd | 1.088560728 | 1.087848813 |
| Final whole-small-FVM velocity error (% U∞) | 0.766375 | 0.746246 |
| Final near-body FVM velocity error (% U∞) | 0.195493 | 0.183527 |
| Final centreline near-wake velocity error (% U∞) | 1.101853 | 1.034184 |
| Final off-axis near-wake velocity error (% U∞) | 0.772103 | 0.762442 |

Whole-FVM and near-body velocity errors decrease by `2.626%` and `6.121%`
relative to their baseline values. Both observed exterior near-wake profiles
improve, but the largest instantaneous relative drag error increases slightly.
Final reference `Cd` remains `1.080395340`; agreement is still far from
roundoff. Final particle population is 28,572, versus baseline 28,441.

![Body-transport comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-twenty-comparison-qualified/comparison.png)

The figure was visually inspected. The
[qualified comparator](compare_body_transport_weighted_3d.py) reconstructs
ten wall-force vectors and the FVM profiles from canonical checkpoints.
It passes 141 independent scalar checks with zero maximum difference,
including the final sampled-cell VPM metric with its native volume weights.
The first comparator incorrectly used an unweighted norm for that additional
metric and stopped before writing a complete result; its source and failure
record remain intact. Correcting the checker changed no simulation data or
acceptance tolerance.

The schedule check confirms 40 body-velocity calls and 40 analytical-gradient
calls at actual RK stages, 20 GBD calls and 80 outer stabilization phases.
Panel strengths remain fixed within each VPM interval. Minimum observed stage
clearance is `0.00412923`; no stage target enters the cube. Direct evaluation
of the final saved sources differs from the runtime profile query by
`6.355e−7 U∞` vector RMS, maximum `5.162e−6 U∞`.

The [complete comparison record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-twenty-comparison-qualified/body-transport-comparison-3d.json)
identifies 526 source/artifact records and retains the unshifted histories,
forces, profiles and PNG/SVG exports. All 799 original frozen files remain
unchanged. This qualifies a useful short-run experimental change; the
body-enabled developed wake and its interaction with more accurate FMM
stage induction remain to be tested before changing production defaults.

An otherwise identical 70-interval body-transport run is now advancing to
physical time `4.0`, matching the existing verified baseline wake window.
Its first 20 intervals have passed the short-prefix repeatability check:
the recorded history exactly reproduces the short run, all 20 intervals
converge, and ten independently reconstructed force vectors and 140 scalar
checks pass with zero maximum difference. The
[fixed-prefix qualification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-long-prefix-through-one-point-five-qualified/long-wake-prefix-verification-3d.json)
covers physical time `0.5–1.5` only; the later body-enabled wake remains
subject to the extended check below.

The long run's first 40 intervals are now independently verified through
physical time `2.5`. All 40 converge; 18 wall-force reconstructions and
276 scalar checks pass, with maximum difference `1.110e-16`. The nine full
reference velocity frames and all 41 reference drag observations match the
same portion of the qualified original baseline bitwise.

| Measurement, physical time 0.5–2.5 | Original particle transport | With body velocity/stretching |
| --- | ---: | ---: |
| Drag-history RMS relative error (%) | 0.803030 | 0.730295 |
| Maximum absolute relative drag error (%) | 1.481333 | 1.490997 |
| Final hybrid Cd | 0.963096021 | 0.962275324 |
| Final whole-small-FVM velocity error (% U∞) | 0.895693 | 0.874970 |
| Final near-body FVM velocity error (% U∞) | 0.432316 | 0.407447 |
| Final centreline near-wake velocity error (% U∞) | 2.063765 | 1.813378 |
| Final off-axis near-wake velocity error (% U∞) | 0.855061 | 0.840842 |

The reference endpoint `Cd` is `0.952218170`. Whole-FVM and near-body
velocity errors decrease by `2.314%` and `5.753%` relative to the original
values. The benefit persists, while the peak instantaneous drag error still
increases slightly and the remaining wake discrepancy is substantial.

The extended [force histories](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-long-prefix-through-two-point-five-figures/forces.png),
[velocity profiles](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-long-prefix-through-two-point-five-figures/profiles.png)
and [velocity-error histories](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-long-prefix-through-two-point-five-figures/velocity-errors.png)
have been visually inspected. They retain the startup force discrepancy and
the growth of the near-wake error; apparent overlap on the profile plots does
not imply roundoff agreement. The figure report's eight recorded source,
verification and export hashes match their files.

The [extended prefix record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-transport-long-prefix-through-two-point-five-qualified/long-wake-prefix-verification-3d.json)
and [original baseline record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/long-wake-prefix-through-four-qualified/long-wake-prefix-verification-3d.json)
retain the observations used above; the original baseline history is
restricted to the same first 40 intervals. Final direct/query velocity
difference is `1.127e-6 U∞` RMS, maximum `8.444e-6 U∞`. Frames beyond
`2.5` remain unqualified while this run continues toward `4.0`. The separate
original-schedule long run continues toward `20.5`.

The [audit record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/body-query-transport-gap-prefix-four/body-query-transport-gap-3d.json)
contains all regional measurements, source hashes and explicit scope limits.
The frozen script and its archived copy are the executed numerical sources.
The repository copy subsequently received import formatting only; the frozen
source and recorded results remain unchanged.
