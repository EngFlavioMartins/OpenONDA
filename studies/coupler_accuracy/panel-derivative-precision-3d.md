# Body-panel derivative precision in the fully 3D coupler

Differencing single-precision body velocities amplifies small numerical changes
in the coupling derivative. In four frozen physical cube fields, its derivative
error is `1.48e-5` to `3.87e-5 U∞/D`, despite velocity errors below `1e-8 U∞`.
Evaluating the same rounded panel geometry and strengths in double precision
reduces derivative error to roughly `1e-11` to `5e-11 U∞/D` inside the existing
single-precision Taichi runtime.

This is a qualified component result. Its scale is relevant to the few-`1e-6`
residual floor in the [interface-iteration experiment](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/interface-iteration-3d.md).
It is much smaller than that experiment's remaining reference boundary errors.
A matched three-interval experiment in a frozen source copy connects this
evaluation error to the iteration floor: all three intervals converge in three
sweeps with promoted queries, versus 9, 7 and a capped 12 with native queries.
The final drag coefficients differ by only `6.28e-8`. The completed longer
promoted-query comparison now converges **all 20 intervals in three sweeps
each**, retaining the drag-history improvement from 2.082% to 0.555% RMS
relative error. No production precision policy has changed.

## Same source field, independent derivative reference

The [precision diagnostic](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/panel_derivative_precision_3d.py)
uses four saved physical body fields on the actual 108-panel cube and 864
coarse coupling faces. Their original double-precision body velocities replay
bitwise. It then rounds the vertices, normals and source strengths to their
single-precision storage values and holds those values fixed for every
evaluation method. The body problem is not solved again.

The reference integrates both the source velocity kernel and its analytical
target directional derivative over each triangular panel. For `R = x-y`,
the directional derivative of `R/|R|³` along unit normal `n` is

```text
n/|R|³ - 3 R (R·n)/|R|⁵.
```

Tensor-product Duffy quadrature at orders 8 and 12 agrees within `1.57e-16`
over the tested velocity and derivative arrays. Targets lie well outside the
body, so this is a smooth surface integral. The analytical triangular kernel
and the independent surface quadrature therefore provide separate evaluations.

The diagnostic uses the current auxiliary difference step
`max(1e-6, 1e-3 h) = 1.25e-4` for this coarse `h=0.125` case. Both methods use
the same true face normals and area weights. The following double-query values
are measured inside a Taichi runtime whose default precision remains `f32`.

| Frozen panel field | Single-query derivative error RMS | Double-query derivative error RMS |
| --- | ---: | ---: |
| Constant-volume source control | 3.87123e-5 | 5.04817e-11 |
| Circulation/LSQ moments | 3.35156e-5 | 2.58596e-11 |
| Native-face moments | 1.47780e-5 | 1.70132e-11 |
| Linear-face moments | 1.54841e-5 | 1.01506e-11 |

All values are in `U∞/D`. Halving the difference step roughly doubles the
single-query error; reducing this step is not a remedy. A separate all-double
runtime shows the expected approximately fourfold truncation-error reduction
when the step is halved. The promoted query inside the single runtime retains
a small additional difference from that all-double calculation; it is not
claimed to be a roundoff-exact derivative.

## Sensitivity to one representable strength increment

Each stored strength is also moved by one representable `float32` increment
toward positive infinity. This is a controlled sensitivity probe, not a
measured change from an advancing renewal or a new Neumann solution. It does
not impose an additional circulation or body-flux constraint on that probe.

| Frozen field | Quadrature derivative response | Single-query response | Double-query response error |
| --- | ---: | ---: | ---: |
| Constant-volume source control | 1.43756e-9 | 2.83649e-6 | 1.11252e-14 |
| Circulation/LSQ moments | 1.10762e-9 | 2.04721e-6 | 7.87474e-15 |
| Native-face moments | 4.38873e-10 | 9.40922e-7 | 4.28715e-15 |
| Linear-face moments | 4.55157e-10 | 8.69324e-7 | 3.44478e-15 |

The single-query response is about 1,850–2,150 times the true response.
The promoted query follows the independently integrated response closely.
This provides a concrete mechanism for small derivative fluctuations even
when the underlying solved panel field changes only slightly.

![Frozen 3D panel derivative precision and sensitivity](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/panel-derivative-precision-3d-qualified.png)

The [diagnostic record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-panel-derivative-precision-coarse-qualified/panel-derivative-precision-3d.json)
contains the geometry, strengths, all evaluated fields, source hashes and
quadrature checks. The [figure's independent field verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/panel-derivative-precision-3d-qualified.json)
recomputes 84 reported metrics with maximum difference `1.36e-20`.

## Advancing qualification and source isolation

The [scoped advancing runner](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_interface_precision_3d.py)
promotes only the auxiliary panel query used by target finite differences.
Stored panel geometry and strengths, the panel solve, complete point-velocity
queries and the particle-stage body operations retain their original precision.
It explicitly requires the original direct 108-panel Neumann body callback,
with no VLM or additional regularized sources. The original method is restored
when the experiment ends.

An initial shared-workspace pair completed, but subsequent verification found
that `source/solvers/fvm/core/solver.py` had changed after it was archived. That
observed edit adds VTK mesh fields. A later three-interval run also failed
while Taichi read a changed stage routine, with `Name "body" is not defined`.
Those attempts are preserved and are not a qualified advancing precision
comparison. No equality gate or numerical tolerance was relaxed.

The [snapshot builder](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/freeze_coupler_workspace.py)
now copies the complete source/package tree, the study scripts and the explicit
cube inputs, verifies that source bytes were stable over the copy window, and
records every copied file. The
[qualified frozen workspace](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/frozen-workspace.json)
contains 786 code files and 13 input files. Its experiments use that copy as
both working directory and `PYTHONPATH`, with a separate Taichi/Numba cache.
Installed dependencies still come from the OpenONDA Python environment.

The [frozen zero/one-sweep verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-one-step-verification.json)
passes: comparison histories, 17 decoded FVM entries, 11 boundary-history
entries and 11 numeric VPM datasets match bitwise. It checks 93 source/artifact
records and ten independent metrics with zero metric difference.

## Matched medium advancing precision pair

The frozen pair uses the same 16,936 small-FVM cells, `h=0.0625`, 108-panel body,
laminar model, physical initial time `0.5`, `dt_FVM=0.01`, coupling interval
`0.05`, and maximum 12 sweeps. The native and promoted runs use identical
captured source bytes; the promoted runner changes only its documented
auxiliary-query evaluation. Each endpoint map replays its own fields and
clocks bitwise. The full-reference histories and saved reference fields match
exactly, as do the two runs' initial comparison records.

| Coupling interval | Native queries: sweeps | Native final derivative residual | Promoted queries: sweeps | Promoted final derivative residual |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9, converged | 6.12196e-10 | 3, converged | 9.66791e-9 |
| 2 | 7, converged | 9.60352e-7 | 3, converged | 8.81209e-9 |
| 3 | 12, capped | 3.06841e-6 | 3, converged | 1.20049e-8 |

Residuals are in `U∞/D`; both normal-velocity and derivative tolerances remain
`1e-6`. The promoted normal residuals at acceptance are about `2e-8 U∞`.
The promoted query reaches the existing criteria earlier; it is not claimed
to have a smaller final residual in every individual interval. Total logical
sweeps decrease from 28 to 9, excluding one replay evaluation per interval.

At physical time `0.65`, the native and promoted drag coefficients are
`1.460416220203181` and `1.4604162830031233`. Whole-small-FVM velocity errors are
`0.00272785137` and `0.00272784191 U∞`; near-body errors are `0.00066224658` and
`0.00066224240 U∞`. These close results support removing the numerical floor
without materially changing this short corrected trajectory. They do not
establish a new improvement in reference accuracy from precision alone.

The [promoted trajectory verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-three-step-verification.json)
checks 192 source/artifact records and 36 independent metrics with zero
difference. The [precision-pair verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-pair-verification.json)
checks all 799 original frozen files and 59 additional native-run metrics,
including an independent final force reconstruction, with maximum difference
`2.22e-16`. The frozen native run's complete comparison prefix and first-map
fingerprints also match the earlier 20-interval native run exactly through
these three intervals.

## Complete 20-interval comparison

Both promoted-query runs finish at physical time `1.5`, with the full FVM
advancing independently through the same 100 steps. The iterated run converges
all 20 intervals in three sweeps each: 60 logical sweeps plus 20 first-map
replays. The largest accepted normal and derivative residuals are
`2.49465e-8 U∞` and `1.81032e-8 U∞/D`, below the unchanged `1e-6` thresholds.
The earlier native-query run required 186 logical sweeps and converged only
12 intervals. Wall-time ratios are not used as a controlled performance claim.

| Metric | Promoted-query original coupling | Promoted-query iterated interface |
| --- | ---: | ---: |
| Drag-history RMS relative error, 20 accepted endpoints | 2.082312% | 0.555068% |
| Final drag coefficient | 1.0577668953 | 1.0885607283 |
| Final signed drag error | −2.094460% | +0.755778% |
| Final whole-small-FVM velocity RMS / U∞ | 0.0076863010 | 0.0076637453 |
| Final near-body FVM velocity RMS / U∞ | 0.0020049769 | 0.0019549317 |
| Final sampled VPM velocity RMS / U∞ | 0.0312452878 | 0.0312425995 |

The full-reference final drag coefficient is `1.0803953399803075`. Final FVM
velocity errors decrease 0.293% over the whole small domain and 2.496% near the
body. The sampled VPM error is essentially unchanged; these samples are at
matched FVM cell centres, not exterior-wake profiles. Early transient velocity
changes need not have the same sign as the final difference.

The [complete verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-twenty-step-verification.json)
checks 243 source/artifact records and 138 independent metrics, with zero
metric difference. It verifies every map replay, unchanged reference histories
and final reference fields, the exact short zero/one-sweep control, and final
wall forces and velocity norms recomputed from saved arrays. The full control
matches the short control's shared prefix. It does not claim a separate
20-interval zero/one-sweep trajectory identity.

![Verified fully converged 3D coupling trajectory](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/precision-twenty-step-trajectory.png)

The plot was visually checked. Interface consistency now has a reliably
converged advancing comparison, while substantial reference error remains.
The subsequent [time-resolution experiment](interface-time-resolution-3d.md)
reduces the exchange/VPM step from `0.05` to `0.01` at fixed FVM step and mesh.
All 100 iterated intervals converge, but its modest common-time drag improvement
comes with 13.49% larger final whole-FVM velocity error. Developed
three-dimensional wake profiles and near-roundoff hybrid agreement remain
outstanding.
