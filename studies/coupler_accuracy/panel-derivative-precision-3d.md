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
An advancing precision comparison is being qualified in a frozen source copy;
no production precision policy has changed.

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

The zero/one-sweep control is being repeated in that workspace before the
three-interval convergence comparison. The required next evidence is exact
control identity, successful first-map replay in every interval, and measured
convergence with the original thresholds. A fully converged longer trajectory
and developed-wake force/profile agreement remain outstanding.
