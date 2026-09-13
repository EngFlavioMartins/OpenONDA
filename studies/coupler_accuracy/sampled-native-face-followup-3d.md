# Observing the VPM field through native FVM faces

Native face sampling improves the frozen boundary comparisons but does not
resolve the advancing coupler's error. On the matched medium 3D cube, changing
only the derivative moves final drag error from −2.0955% to −2.0769%, with
almost unchanged velocity errors. Changing both boundary observations improves
final small-FVM velocity errors by about 3% but worsens drag to −2.2373%.
Neither experiment qualifies as a production solution to the requested
simultaneous force/profile agreement.

The frozen comparison establishes a distinct error source: the FVM face
stencil and a continuous derivative give materially different observations of
the same reconstructed velocity. Native sampling improves both boundary
comparisons in all 15 tested source/body combinations. For the latest
quadratic Γ-and-M source at 6,912 body panels, normal-velocity error falls
26.5% and tangential-derivative error falls 16.3%. Those isolated gains do not
predict the advancing result by themselves.

This follows the [wall-informed reconstruction study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/boundary-informed-reconstruction-3d.md).
The small FVM domain, reconstructed sources, Gaussian complements, 68 exterior
particles and solved body strengths are unchanged. All meshes and velocity
components are fully three dimensional.

## What the diagnostic changes

The [sampling component](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native_face_velocity_sampling_3d.py)
identifies both cells adjacent to each coupling face and all cells needed for
their native Gauss gradients. The coarse physical mesh requires 3,088 velocity
samples to complete 1,592 cell gradients at the 864 coupling faces. The
velocity at every sample comes from the same reconstructed source field and
saved source-panel solution. Exterior FVM solution values are used only as
comparison data; they do not enter the reconstructed source or sampled field.

For a cut face oriented outward from the small FVM, let `d` connect its inner
and outer cell centres, `n` be its unit normal, and `Gf` be the native weighted
interpolation of their Gauss velocity gradients. The native unit-diffusivity
trace can be written as

```text
gn = (u_outer - u_inner)/(n·d) + (n - d/(n·d))·Gf
gt = (I - n nᵀ) gn.
```

The runner calls the actual FVM Gauss and diffusion routines. Its independent
verifier accumulates face contributions separately and evaluates the formula
above. Selected gradient cells must have entirely interior stencils; any
dependence on a missing physical-boundary observation is rejected. Remote
placeholder values therefore cannot influence a selected trace.

The native face velocity is the existing FVM interpolation of the two sampled
cell values. Both the point and native normal-velocity observations receive
the same public coupling mass-flux correction. This is a velocity observation,
not an advancing pressure-corrected Rhie–Chow flux. The frozen comparison
retains the earlier face-area convention for its normal-velocity metrics;
the tangential derivative uses the true native unit normal throughout.

## Frozen physical results

At 6,912 body panels, all values refer to exactly the same field within each
row. Only its boundary observation changes:

| Source | Point normal-velocity error / U∞ | Native normal-velocity error / U∞ | Continuous derivative error [U∞/D] | Native derivative error [U∞/D] |
| --- | ---: | ---: | ---: | ---: |
| Constant native volume | 0.00395007 | 0.00365565 | 0.02103324 | 0.01780741 |
| Linear-face M, native Γ | 0.00351012 | 0.00270285 | 0.02167887 | 0.01830796 |
| Quadratic cell M, native Γ | 0.00346048 | 0.00264961 | 0.02160728 | 0.01825225 |
| Quadratic wall M, native Γ | 0.00362318 | 0.00307228 | 0.02194265 | 0.01868244 |
| Quadratic wall Γ and M | 0.00344493 | 0.00253297 | 0.02110576 | 0.01767202 |

![Same field observed continuously and through native faces](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-sampled-native-face.png)

The Γ-and-M source's derivative error decreases 16.6%, 16.2% and 16.3% at
108, 1,728 and 6,912 panels. Its normal-velocity error decreases 27.1%, 26.2%
and 26.5%. These are reductions relative to the continuous/point observation
of that same source at that same body resolution.

The largest derivative change occurs on the downstream `xmax` face. At
6,912 panels, its error falls from 0.04145264 to 0.03076134 U∞/D. The four
transverse faces change much less, retaining roughly 0.0151 U∞/D error each.
The native nonorthogonal correction has RMS 0.00010413 U∞/D, compared with
0.00515821 U∞/D for the full change of observation. Finite cell-to-cell
sampling accounts for most of this difference.

The error contributions are correlated. With `ec` the continuous-observation
error and `delta` the change to native observation, the verifier checks
`||ec+delta||² = ||ec||² + 2<ec,delta> + ||delta||²` using face-area weights.
Subtracting RMS magnitudes would not identify an independent error budget.

The [complete frozen experiment](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_sampled_native_face_3d.py)
took about 110 seconds. The reference derivative replays exactly, and every
previous boundary point velocity replays within `4.45e-16`. All source-panel
velocities use the exact production triangular kernel, without a new body
solve or far-field grouping.

Four [sampling tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_native_face_velocity_sampling_3d.py)
pass: arbitrary velocities on warped cells versus the full native operator;
affine recovery in three directions on skew cells; second-order convergence
to an analytical smooth 3D derivative; and rejection of missing physical
boundary data and empty patches. Changing every unsampled cell and boundary
value leaves all selected gradients bitwise unchanged.

The [frozen verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_sampled_native_face_3d.py)
checks 42 source records and independently recomputes 721 metrics, with maximum
difference `1.12e-16`. An independent Gauss/flux reconstruction differs by at
most `1.28e-15`. The numerical record is in
[sampled-native-face-verification.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/sampled-native-face-verification.json).

## Advancing qualification

A scoped [boundary adapter](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/experimental_sampled_face_boundary.py)
now supports two explicit experiments. `native_gradient` changes only the
derivative and retains the original point-velocity query. `native_both` also
returns native interpolated face velocity. Both sample velocity from the
current VPM solver, including its freestream and body response. They retain
the current transfer, pressure policy and convection assembly.

The [live runner](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_sampled_face_trial_3d.py)
uses the same medium laminar seed and small FVM mesh as a fresh control:
53,752 full cells, 16,936 inherited small cells, 0.0625D wall and particle
spacing, FVM steps of 0.01 and coupling steps of 0.05. Its 3,456 boundary
faces require 13,024 VPM evaluation points for 6,632 native cell gradients.
These exterior evaluation points do not enlarge the solved FVM domain.
The experiment currently requires the full mesh's geometric stencil; a
general coupler would need to construct a suitable exterior sampling stencil
from its own available geometry.

Three [adapter tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_experimental_sampled_face_boundary.py)
verify the two observation modes, query flags and sample locations, rejection
of different faces/orientations, and restoration of the original method after
an exception. The experiment remains under studies and does not add a
production option.

All three runs completed 20 coupling steps and 100 FVM steps, reaching physical
time 1.5. The live VPM uses the current Gaussian particles, FMM/RK2/GBD in f32
and 108 source panels. The richer volume/moment sources and finer f64 body
resolutions from the frozen experiment have not been introduced into live
transfer here.

| Final measurement | Current continuous trace | Native derivative, point velocity | Native derivative and face velocity |
| --- | ---: | ---: | ---: |
| Whole small-FVM velocity RMS / U∞ | 0.007686214 | 0.007689203 | 0.007466694 |
| Near-body FVM velocity RMS / U∞ | 0.002004944 | 0.002001593 | 0.001948384 |
| VPM velocity RMS at independent cells / U∞ | 0.031245271 | 0.031245950 | 0.031243935 |
| Drag coefficient | 1.057755151 | 1.057956950 | 1.056223586 |
| Relative drag error | −2.09555% | −2.07687% | −2.23731% |
| RMS drag-coefficient error over the 20 advancing samples | 0.026326427 | 0.026251883 | 0.027871579 |
| Final particle count | 28,458 | 28,460 | 28,450 |

The full reference has `Cd=1.0803953399803075`. The derivative-only variant
reduces final drag-error magnitude by 0.89%, or 0.01868 percentage points. Its
near-body velocity error falls 0.17%, while whole-volume and VPM errors rise
slightly. Over the entire recorded interval, drag-error RMS improves only
0.28%.

Using both native observations reduces final whole-volume and near-body FVM
errors by 2.86% and 2.82%, while VPM error barely changes. Its final drag-error
magnitude rises 6.76%, and the drag-error RMS over the interval rises 5.87%.
Near-body velocity also worsens over part of the earlier trajectory. Selecting
only the final velocity improvement would miss these force and transient
tradeoffs.

![Advancing matched medium cube comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-sampled-face-live.png)

An independent reconstruction of the final wall forces from each saved FVM
restart recovers all three hybrid drag coefficients exactly. It rebuilds the
Gauss cell gradient, corrects the wall-normal derivative to the saved no-slip
trace, and integrates pressure plus the deviatoric viscous stress. The force
sampler's reported coefficient is not used in this reconstruction.

| Final drag contribution | Current | Native derivative | Native derivative and velocity |
| --- | ---: | ---: | ---: |
| Pressure | 1.004660690 | 1.004864317 | 1.003167836 |
| Viscous | 0.053094462 | 0.053092633 | 0.053055750 |

Most of the change in drag comes through the pressure force. This identifies
the measured response; it does not establish a new pressure-boundary fix.

The [live verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_sampled_face_trials_3d.py)
checks 81 source records, reproduces 87 numerical comparisons exactly, and
independently replays the initial and final sampled gradient stencils to
`1.85e-15`. Each variant made 41 boundary queries. The initial comparisons,
full-reference force histories and final reference velocity arrays are
bitwise identical across all three runs. The saved hybrid restart velocities
also match their final comparison arrays bitwise.

The fresh control differs slightly from the earlier live baseline: the
largest historical change is `5.54e-7` in hybrid Cd and `9.27e-9 U∞` in the
whole-volume velocity RMS. The experiments use the fresh control, and these
small differences are recorded without assigning an unverified cause.
Seven new component/adapter tests pass. The combined verification record is
[sampled-face-trials-verification.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/sampled-face-trials-verification.json).

## Consequence for the next transfer experiment

Native sampling is now a qualified diagnostic and a tested live boundary
variant. It does not remove the approximately 2% force discrepancy with the
current transfer. Both variants remain scoped to studies. This result does
not rule out combining native observations with an improved source
representation, which the live runs have not tested.

The next source comparison should resolve the physical medium FVM cell-value
interpretation and the Γ/M budget of the overlap before particle emission is
changed. In particular, multiplying a cell curl by a varying overlap weight
does not retain the shared-face cancellation of a curl formed from weighted
velocity traces. A proposed conservative update must state which circulation
and first-moment budgets it preserves and verify them independently; the
preceding frozen Γ-and-M candidate has not yet done that. Developed tutorial
force/profile agreement and near-roundoff equivalence remain unachieved.

Reproduction uses the OpenONDA environment, `PYTHONPATH=.` and single-threaded
BLAS, with new output directories:

```sh
python studies/coupler_accuracy/cube_sampled_native_face_3d.py \
  --output /private/tmp/cube-sampled-native-face
python studies/coupler_accuracy/cube_coupled_trial.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --particle-spacing 0.0625 --steps 20 --substeps 5 \
  --output /private/tmp/cube-sampled-face-control
python studies/coupler_accuracy/cube_sampled_face_trial_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --mode native_gradient --output /private/tmp/cube-sampled-face-gradient
python studies/coupler_accuracy/cube_sampled_face_trial_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-medium-laminar-oracle \
  --mode native_both --output /private/tmp/cube-sampled-face-both
```

The verifiers target the qualified result locations used in this report;
they do not silently substitute the reproduction directories above.
