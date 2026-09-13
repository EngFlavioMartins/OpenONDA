# Prescribed wall velocity in the reconstruction

Using the known wall velocity in the quadratic cell fit reduces the medium
manufactured thin-field error and improves the frozen physical cube when its
circulation and first moments are updated together. At 6,912 body panels, that
physical candidate lowers near-body velocity error by 13.0%, coupling normal
velocity error by 1.9%, and tangential-derivative error by 2.6% relative to the
previous linear-face moment source. These remain component results, not the
requested advancing hybrid/reference force and velocity agreement.

This continues the [quadratic reconstruction and panel study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/moment-panels-and-quadratic-3d.md).
The physical FVM domain remains approximately `[-1.5,1.5]^3`, with the same
2,840 cells inherited from the full reference and the same 68 exterior seed
particles. All meshes, fields, stencils and induction kernels are fully 3D.
No production coupling settings or emission method change in this follow-up.

## What wall data changes

The previous fit used neighbouring cell velocity only. The new
[component](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native_boundary_quadratic_3d.py)
adds prescribed velocity on selected boundary faces whose owners lie within
the same cell-neighbour stencil. It retains the nine nonconstant quadratic
modes, the same two-ring cell stencil, and the distinction between point and
cell-average input. Only the cube boundary is supplied in the flow experiments.

The objective adds

```text
Σnearby prescribed faces ∫face |u_polynomial(y) − u_boundary(y)|² dA
                         / |Cface − ccell|²
```

to the previous distance-weighted cell mismatch. `Cface` is the actual face
fan's area centroid. The existing cell weights are `hcell²/dcell²`, with
`hcell=Vpoly^(1/3)`. Both parts therefore have velocity-squared units. There is
no adjusted wall-penalty coefficient or tuning against target velocities.

Three-point tensor Gauss–Jacobi quadrature on each fan triangle integrates the
quadratic residual's squared magnitude. Face-area weights make the objective
independent of the number of quadrature samples. The fit is soft: prescribed
boundary velocity is an observation, while the shared-face weak-curl integrals
still use the supplied boundary trace directly. A better polynomial fit to wall
data does not imply that the later projected induction satisfies no slip.

All nine modes must already be observable from the full 3D cell stencil. Wall
observations cannot turn a two-dimensional cell stencil into an accepted fit.
Stencils remain within two rings on both meshes. Maximum design condition
numbers stay below 9.4. Wall data enters 1,536 coarse cells and 4,704 medium
cells. Including the neighbouring shared traces, only 1,832 and 5,432 source
cells can change; all other Γ and M reproduce the previous reconstruction
exactly.

Five new [tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_native_boundary_quadratic_3d.py)
pass: general quadratic recovery on warped cells for point and average input;
quadrature-order invariance with inconsistent cell observations; independently
integrated reduction of the prescribed-wall mismatch with an unchanged bulk
fit; and rejection of a two-dimensional cell stencil. The noisy-observation
checks verify that the boundary data has an effect, rather than merely passing
an exact polynomial that the cell-only fit already recovers.

## Manufactured results

The coarse and medium experiments retain all 17,592 and 53,752 source cells,
respectively, and the same 2,048 evaluation targets. The input fields have exact
zero velocity on the cube and variation in all three directions. Point and
exact-average inputs are separate measurements; neither method receives exact
derivatives, exact vorticity moments or target velocities.

![Wall observations in the manufactured reconstruction](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-boundary-quadratic-manufactured.png)

Near-body velocity RMS normalized by the manufactured reference speed:

| Reconstruction | Broad coarse | Broad medium | Thin coarse | Thin medium |
| --- | ---: | ---: | ---: | ---: |
| Quadratic, point input, cell observations | 0.037387 | 0.007087 | 0.068112 | 0.033003 |
| Quadratic, point input, cell and wall observations | 0.035719 | 0.007031 | 0.076271 | 0.021296 |
| Quadratic, average input, cell observations | 0.042836 | 0.009026 | 0.069443 | 0.018381 |
| Quadratic, average input, cell and wall observations | 0.042212 | 0.008943 | 0.077126 | 0.020590 |
| Native Γ with M from point and wall observations | 0.046076 | 0.019373 | 0.072719 | 0.020863 |

With point input, the medium thin-field near-body error falls **35.5%**, and
wall velocity error falls **38.5%**, relative to the preceding quadratic fit.
Its relative circulation error falls from 22.50% to 19.93%, and the error in
affine vorticity variation from 62.80% to 37.58%. The broad medium near-body
gain is retained, with a further 0.8% reduction, although its wall error rises
11.9%.

The average-input result remains mixed. Medium thin-field near-body error rises
12.0%, while wall error falls 12.9%. On the coarse mesh, both input types worsen
the thin-field near-body error while lowering wall error. The source
reconstruction therefore still has significant mesh and input dependence;
wall observations do not give a universal accuracy improvement. The complete
coarse and medium runs took about 111 and 333 seconds.

## Frozen physical comparison

The [physical runner](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_boundary_quadratic_overlap_3d.py)
uses the same full-FVM snapshot at `t=0.5`, restricted to the existing small FVM
mesh. Its current cell values are treated as point inputs to the candidate;
this does not assert that a computed FVM value equals an exact physical point
sample. No exterior FVM cells or derivative oracle enters either cell fit.
The prescribed cube velocity is zero. Outer placeholder face values cannot
touch active source faces, which is checked topologically.

The first four source states preserve **every original circulation**. They
compare constant volumes, the preceding linear-face M, quadratic M from cell
values, and quadratic M from cell and wall values. The fifth state also uses
the shared Stokes sum from the cell and wall fit for Γ in the volume portion of the overlap.
All retain the same taper, Gaussian complements and exterior particles.

Only the changed induction and its body response are recomputed. The actual
constrained Neumann solve is linear, so its response to each correction is
added to the qualified constant-volume state. The two prior physical controls
are explicitly reproduced at each body resolution, to at most `1.13e-14` in
any velocity component. Every source-panel velocity
evaluation uses the exact triangular kernel with far-field grouping disabled.

At 1,728 body panels:

| Source | Near-body velocity RMS / U∞ | Boundary normal-velocity RMS / U∞ | Tangential derivative RMS (U∞/D) | Unused-layer velocity RMS / U∞ |
| --- | ---: | ---: | ---: | ---: |
| Constant volume | 0.170432 | 0.00396118 | 0.02103697 | 0.00419364 |
| Previous linear-face M, native Γ | 0.122358 | 0.00350122 | 0.02166962 | 0.00306756 |
| Quadratic M from cells, native Γ | 0.116222 | 0.00344949 | 0.02159695 | 0.00297770 |
| Quadratic M from cells and wall, native Γ | 0.113829 | 0.00363338 | 0.02193581 | 0.00357490 |
| Quadratic Γ and M from cells and wall | 0.106814 | 0.00342342 | 0.02108744 | 0.00257174 |

Changing only M improves near-body velocity but does not improve both coupling
components. Updating Γ and M together improves all three listed velocity
regions, both coupling errors and both wall velocity components relative to
the previous linear-face moment source at this panel count. Nevertheless, its
tangential-derivative error is still slightly greater than the constant-volume
control. Substantial error remains.

The same Γ-and-M candidate was checked at three body resolutions, with its
incident source field bit-for-bit identical at all common targets:

| Body panels | Near-body velocity RMS / U∞ | Boundary normal-velocity RMS / U∞ | Tangential derivative RMS (U∞/D) | Wall-normal RMS / U∞ | Wall-tangential RMS / U∞ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.093926 | 0.00376581 | 0.02128975 | 0.115910 | 0.093758 |
| 1,728 | 0.106814 | 0.00342342 | 0.02108744 | 0.068263 | 0.202979 |
| 6,912 | 0.105672 | 0.00344493 | 0.02110576 | 0.052456 | 0.210235 |

![Physical comparison at three body resolutions](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-boundary-quadratic-physical.png)

At the finest body resolution, the improvements relative to the previous
linear-face moment source are 13.0% in near-body velocity, 15.9% in the unused
outer layer, 8.6% in the wake sample, 1.9% in normal boundary velocity, and 2.6%
in its tangential derivative. Wall normal and tangential errors fall 5.9% and
4.1%. Relative to the constant-volume control, near-body error falls 38.3% and
normal boundary error falls 12.8%, but derivative error rises 0.34% and wall
penetration rises 8.2%. The reference used for each percentage matters.

The initial 108-panel apparent near-body accuracy does not survive body
refinement unchanged. The two finer resolutions provide a more stable boundary
comparison, while wall errors still depend on panel count. The three complete
physical runs took approximately 101, 127 and 328 seconds. All derivative
step-halving component differences remain below `1e-7`, and every body source
flux remains below `3e-15`.

The circulation change is material. Its L1 magnitude is `1.78711`, versus
`9.15188` for the original near-cell circulation and `8.70610` for its weighted
volume portion: **19.5% and 20.5%**, respectively. Its signed vector sum is
approximately `(1.3793e-6, −3.0575e-5, 3.2746e-5)`, with magnitude `4.4822e-5`.
The small signed sum does not make the local changes small. The fifth state
tests a changed curl/transfer definition; it is not a circulation-preserving
particle-emission scheme. That distinction must be resolved before using it in
an advancing coupler.

## Validation and remaining work

The [verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_boundary_quadratic_3d.py)
recomputes source and velocity errors, checks the unaffected-source identity,
replays all six boundary derivative stencils independently, reconstructs the
normal-flux correction, checks Γ/M budgets and body source flux, and verifies
**106 source records** and the five passing tests. It independently reproduces
**426 metrics**, with maximum difference zero; the independent derivative
stencil replay also has maximum difference zero. Its numerical record is saved in
[boundary-quadratic-verification.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/boundary-quadratic-verification.json).

The first two physical attempts stopped at a missing argument in the study's
wall-RMS calculation after completing source induction. They produced no
qualified physical report. The corrected runs use explicit uniform wall
weights and repeat the complete experiment. The initial directories carry an
`incomplete_metric_evaluation` status and are excluded from comparisons.

Before a production change, the representation must be checked on the physical
medium mesh and its interpretation of FVM cell values must be resolved. Another
useful separation is to observe the induced field through the actual FVM
internal-face stencil, alongside its continuous derivative. The preceding
[native-face investigation](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native-face-boundary-followup-3d.md)
already showed that these are different discrete observations of a reference
field. A live candidate must then preserve a defined circulation/impulse budget
and pass the advancing matched-mesh force/profile comparison. The current
results do not yet meet that goal.

The proposed same-field native-face separation has since been
[completed and tested in the advancing medium coupler](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/sampled-native-face-followup-3d.md).
It improves the frozen boundary measurements but does not remove the live
force discrepancy. The source-value interpretation and conservative Γ/M
transfer qualification above remain outstanding.
