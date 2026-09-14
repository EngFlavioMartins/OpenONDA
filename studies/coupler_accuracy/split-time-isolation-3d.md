# RK/GBD substeps with fixed FVM exchange in the 3D cube

Finer particle advection/stretching and diffusion/remapping together do not
improve overall agreement in this short matched comparison. With exchange
and accepted renewal held at `0.05`, five native RK2/GBD pairs of `0.01`
raise drag-history RMS error from `0.555068%` to `0.583542%`. Final
whole-small-FVM velocity error rises `1.079%` relative to its baseline value.
Centreline near-wake velocity error increases, while the off-axis error
decreases slightly. This experiment is not selected as an accuracy fix.

The independent full FVM has 53,752 cells; the small FVM has 16,936 cells in
approximately `[-1.5,1.5]^3` around the unit cube. Both use identical native
near-body cells, `h=0.0625`, `dt_FVM=0.01`, and laminar `ν=0.001`. The
fully 3D trajectory covers physical time `0.5–1.5` without reference feedback.

## What changes and what the comparison establishes

The [scoped runner](cube_split_subcycling_3d.py) repeats the native sequence
RK2 then GBD five times per exchange. Each RK call receives its physical
substep start time. The canonical VPM clock commits once; FVM exchange,
accepted renewal and the four outer stabilization phases remain at `0.05`.
This changes GBD frequency and its splitting with RK together. It does not
separate diffusion, remapping, pruning and splitting from one another.

The table combines the original schedule, the completed
[RK-only experiment](inviscid-time-isolation-3d.md), this candidate, and the
previous [complete exchange refinement](interface-time-resolution-3d.md).
Force statistics use the same 20 physical endpoints, excluding the common
initial state. Final velocity errors are vector RMS divided by `U∞`,
expressed as percentages; they are not relative changes in an existing error.

| RK dt | GBD dt | Exchange / outer stabilization dt | Drag-history RMS error (%) | Final whole-FVM velocity error (% U∞) | Final near-body velocity error (% U∞) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.05 | 0.05 | 0.555068 | 0.766375 | 0.195493 |
| 0.01 | 0.05 | 0.05 | 0.555463 | 0.766615 | 0.195594 |
| 0.01 | 0.01 | 0.05 | 0.583542 | 0.774644 | 0.204095 |
| 0.01 | 0.01 | 0.01 | 0.488147 | 0.869738 | 0.206487 |

The complete `0.01` exchange case has an `11.009%` first-interval drag
error at `t=0.51`; its RMS over all 100 endpoints is `1.205067%`. The
common-time statistic must not hide that pulse. Both statistics are retained.

Refining RK alone produces almost no change. Refining RK and GBD at fixed
exchange produces a smaller whole-FVM change than refining the complete
exchange. The remaining contrast increases whole-FVM velocity error by
`12.276%` relative to the finer-RK/GBD case. It involves exchange, renewal,
boundary updates and outer stabilization frequency; these interacting errors
cannot be assigned additive percentages from this experiment.

## Profiles and forces

At `t=1.5`, hybrid `Cd` is `1.089464066`, compared with the unchanged
reference `1.080395340` and original iterated hybrid `1.088560728`.

| Final exterior near-wake line | Original RK/GBD | Finer RK/GBD |
| --- | ---: | ---: |
| Centreline vector RMS error (% U∞) | 1.101853 | 1.307890 |
| Off-axis vector RMS error (% U∞) | 0.772103 | 0.747708 |

Near wake means `1.5 < x ≤ 4`, with `z=0`; the off-axis line has `y=0.75`.
These are samples of a fully 3D solution. Final particle counts are 28,441
and 28,167. Direct evaluation of the candidate's saved sources differs from
the runtime profile query by `6.791e−7 U∞` vector RMS, maximum `7.523e−6 U∞`.
It does not remove the reference mismatch.

![Fixed-exchange RK/GBD comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/split-subcycling-five-twenty-comparison/comparison.png)

The figure was visually inspected. It shows physical composite profiles,
using FVM inside the small box and VPM outside. The numerical comparison
also retains matched FVM sampling-stencil metrics. The older finer-exchange
experiment did not save exterior profile frames; none are invented for it.

## Verification

The one-substep wrapper reproduces the original three-interval control
bitwise: histories, six comparison arrays, 17 FVM entries, 11 boundary-history
entries and 11 numeric VPM datasets. Eight independently reconstructed wall
forces at its four saved times agree exactly. Its profile figure was inspected.

The [schedule verifier](verify_split_schedule_3d.py) checks all 100 RK calls,
200 stages, 100 GBD calls and 80 outer stabilization phases in the candidate.
Every GBD call retains the production correction limit `0.08` and normalized
residual limit `1e−5`. Maximum reported correction fraction is `4.391e−4`;
maximum reported circulation, linear-impulse and angular-impulse residuals
are `8.169e−10`, `5.028e−10` and `1.356e−10`. This checks the recorded
diagnostics; it does not independently reconstruct intermediate GBD grids.

The [comparison verifier](compare_split_subcycling_3d.py) passes 140 scalar
checks with zero maximum difference, including ten independently reconstructed
wall-force vectors. All 20 interface intervals converge in three sweeps,
and their first maps replay 24 recorded state/clock entries bitwise. Initial
source fields and saved full-reference fields match the baseline exactly.
All 799 original frozen source/input files remain unchanged. The
[complete comparison record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/split-subcycling-five-twenty-comparison/split-subcycling-comparison-3d.json)
identifies 1,013 source/artifact records and the PNG/SVG exports.

The candidate preserves the selected mode's omission of the body field from
particle stages. The [body-stage investigation](body-query-and-transport-3d.md)
is a separate controlled path. Neither completed time-refinement experiment
demonstrates the requested force/profile agreement or a developed wake.
