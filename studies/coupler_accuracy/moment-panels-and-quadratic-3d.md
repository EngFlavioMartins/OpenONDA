# Body resolution and quadratic velocity reconstruction

The subsequent [study using prescribed wall data](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/boundary-informed-reconstruction-3d.md)
adds wall velocity to the cell fit and tests its circulation and moment changes
on the manufactured meshes and the frozen physical cube.

Refining the body surface reduces penetration but leaves substantial coupling
errors. Separately, a quadratic reconstruction improves the broad manufactured
field on the medium mesh, while worsening the thin field with point inputs.
These results narrow the remaining reconstruction problem; they do not establish
the requested advancing hybrid/reference force and velocity agreement.

This follows the [first-moment reconstruction study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/reconstructed-moments-followup-3d.md).
All fields, kernels, stencils and meshes remain fully three-dimensional. The
physical comparison retains the same approximately `[-1.5,1.5]^3` FVM domain,
2,840 cells, cell circulation, Gaussian complements and 68 exterior particles.
No production coupling settings change in this follow-up.

## Isolating the body response

The physical source field is frozen at the same `t=0.5` snapshot as the previous
overlap study. Each original cube triangle is subdivided into four children,
giving 108, 432, 1,728 and 6,912 panels. The surface area remains 6 and the enclosed
volume remains 1. All 8,596 evaluation targets remain identical, including the
1,536 independent wall samples and every coupling derivative offset.

The volume and Gaussian induction at these targets is held bit-for-bit fixed
for each source state. Only induction at the new panel centres and the resulting
body source strengths are recomputed. Every level uses the actual f64 Neumann
panel solver and the exact triangular source velocity kernel. The panel count
threshold for far-field grouping is set above the active panel count; its
recorded evaluation fraction must be zero. Thus the refinement does not also
change the velocity evaluation approximation.

The original 108-panel fields reproduce to `2.22e-16` maximum component
difference. Two earlier incomplete baseline attempts combined circulation and
moment contributions before summation and differed by up to `1.21e-12` near the
wall. Restoring the original order—constant-volume plus Gaussian induction,
then the zero-circulation moment correction—resolved that discrepancy without
loosening the reproduction threshold. Those directories are explicitly marked
as superseded and are excluded from the comparison.

![Body panel refinement](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-moment-panel-resolution.png)

For the linear-face moment source from the preceding study:

| Body panels | Near-body velocity RMS / U∞ | Boundary normal-velocity RMS / U∞ | Tangential derivative RMS (U∞/D) | Wall-normal RMS / U∞ | Wall-tangential RMS / U∞ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.111809 | 0.00364411 | 0.02179311 | 0.114415 | 0.102211 |
| 432 | 0.119191 | 0.00351090 | 0.02164335 | 0.098088 | 0.173851 |
| 1,728 | 0.122358 | 0.00350122 | 0.02166962 | 0.071222 | 0.211136 |
| 6,912 | 0.121470 | 0.00351012 | 0.02167887 | 0.055716 | 0.219168 |

Relative to 108 panels, the finest body reduces this source's wall penetration
by 51.3%, but increases near-body velocity error by 8.6%. The normal coupling
error improves by 3.7%; the derivative error changes little. Tangential wall
velocity increases substantially. The source-panel boundary condition imposes
normal velocity, so reducing penetration does not itself impose no slip.

The last refinement's **field differences**, rather than only differences
between error norms, are:

| Fixed source | Δ near-body velocity RMS / U∞ | Δ boundary normal velocity RMS / U∞ | Δ boundary tangential derivative RMS (U∞/D) |
| --- | ---: | ---: | ---: |
| Constant volume | 0.002293 | 0.00003190 | 0.00004707 |
| Circulation-gradient moments | 0.002029 | 0.00001328 | 0.00002250 |
| Native-face moments | 0.004422 | 0.00011591 | 0.00017444 |
| Linear-face moments | 0.003816 | 0.00006446 | 0.00009736 |

For linear-face moments, the final boundary field changes are about 1.8% and
0.45% of the remaining normal-velocity and derivative errors. This is evidence
that these boundary discrepancies are not being removed by body refinement.
It is not a rigorous bound on the uncomputed panel-discretization error. The
independent wall measurements still have appreciable resolution dependence.

The area-weighted discrete Neumann collocation residual is also saved. It is
distinct from wall penetration between collocation points. The actual solver
minimizes its equations subject to zero net source flux; a nonzero equation
residual is not by itself evidence of a failed constrained solve. All body
source fluxes remain below `2.6e-15`. The centred and both one-sided boundary
derivative estimates pass step halving; the largest component change is below
`1.01e-7`. The four runs took approximately 3, 8, 32 and 175 seconds.

## Recovering curvature from cell velocity

The new [quadratic reconstruction component](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native_quadratic_moments_3d.py)
fits nine nonconstant polynomial modes per velocity component from neighbouring
cells:

```text
u_i(y) = U_i + r · G_i + 0.5 (r r − C_i) : H_i,
r = y − c_i.
```

For point input, `c_i` is the stored FVM centre and `C_i=0`. For average input,
`c_i` is the true polyhedral centroid and `C_i=S_i/V_i`, where `S_i` is its exact
second central volume moment. The two definitions preserve the supplied point
value or cell average respectively. They are not interchangeable.

The weighted SVD uses neighbours within two face-adjacency rings. A third ring
is available when the first stencil lacks rank or has excessive condition
number; a stencil that still lacks all nine 3D modes is rejected. None of the
actual coarse or medium cells needs the third ring. Maximum design condition
numbers are about 9.32 for point input and 9.39 for average input. Stencils
contain 9–42 neighbours on the coarse mesh and 9–64 on the medium mesh.

The polynomial is integrated over the actual cell to estimate its velocity
integral. This includes the Hessian contribution missing from the previous
linear estimate. Shared face polynomials are formed by weighted averaging of
the owner and neighbour traces. The weak Stokes and first-moment identities
then give circulation and first vorticity moments. Exact triangle second and
third moments integrate these quadratic traces over the same warped face fans.
The existing affine-vorticity induction kernel consumes the resulting Γ and M.

Prescribed zero cube velocity is used for the boundary face integrals. The
current **cell least-squares fit uses neighbouring cell values only**; it does
not incorporate prescribed wall values as observations or constraints. The
outer manufactured field is exponentially small and is represented by a
constant supplied trace per boundary face. General variable VPM boundary
polynomials are not qualified by these experiments.

Five additional [tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_native_quadratic_moments_3d.py)
pass. They cover exact quadratic reconstruction from point and average inputs
on warped 3D cells, constant preservation, cell integrals checked against
independent signed-tetrahedron quadrature, weak Γ and M checked against direct
volume integration of curl, shared-trace global budgets, and rejection of a
two-dimensional stencil. These are component consistency tests, not claims of
small flow error at finite mesh spacing.

## Manufactured accuracy on unchanged meshes

The same 17,592-cell coarse and 53,752-cell medium meshes, all source cells,
256 near-body targets, 256 outer-layer targets and 1,536 wall targets are used.
Point and exact-average velocity inputs are separate controls. Neither receives
exact derivatives, exact first moments or target velocities. No source pruning,
core-radius tuning or target fitting is applied.

![Quadratic moment reconstruction](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-quadratic-moment-reconstruction.png)

Near-body velocity RMS normalized by the manufactured reference speed:

| Reconstruction | Broad coarse | Broad medium | Thin coarse | Thin medium |
| --- | ---: | ---: | ---: | ---: |
| Previous linear faces, point input | 0.037038 | 0.019943 | 0.068270 | 0.031198 |
| Quadratic faces, point input | 0.037387 | 0.007087 | 0.068112 | 0.033003 |
| Previous linear faces, average input | 0.039468 | 0.009704 | 0.069642 | 0.018555 |
| Quadratic faces, average input | 0.042836 | 0.009026 | 0.069443 | 0.018381 |
| Native Γ with quadratic point-input M | 0.045178 | 0.019261 | 0.069725 | 0.035201 |
| Previous exact Γ and M control | 0.012634 | 0.003478 | 0.052435 | 0.009570 |

On the medium broad field, quadratic point reconstruction lowers near-body
error by **64.5%** and wall velocity error by **32.4%** relative to linear point
reconstruction. Much of the new gain depends on its circulation: retaining
native Γ with exactly the same quadratic M gives `0.019261`, versus `0.007087`
when the quadratic face trace also supplies Γ. The relative near-body Γ error
is 2.71%, versus 12.17% for the native point-velocity curl. The corresponding
moment-variation error changes only slightly from the linear reconstruction,
from 28.81% to 28.42%.

For the medium thin field, point-input near-body error instead **increases by
5.8%**, and wall error by **27.0%**. Quadratic Γ has 22.50% relative error and
its affine moment variation has 62.80% relative error. The previous linear
point reconstruction had 21.13% and 59.12%, respectively. Both are far from the
exact Γ,M control; polynomial exactness and a well-conditioned stencil do not
remove the remaining reconstruction error at this mesh spacing.

Average-input near-body improvements on the medium mesh are smaller: 7.0% for
the broad field and 0.9% for the thin field. Thin-field wall error still rises
by 6.9%. The coarse results show little benefit overall. These measurements do
not support treating quadratic reconstruction as a universal replacement for
the existing transfer. The coarse and medium runs took about 109 and 327 seconds.

## Verification and next decision

The [independent verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_moment_panels_and_quadratic_3d.py)
checks **118 source records** against their archived and current hashes and
recomputes **440 reported metrics**, with maximum difference zero. It also
replays all six derivative formulas independently, checks the unchanged
incident fields across body resolutions, verifies the geometric/source-flux
constraints and reconstructed global Stokes/moment budgets, and checks the five
passing tests. The [verification record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/moment-panels-and-quadratic-verification.json)
and both figures are saved alongside the raw source and target arrays.

The next reconstruction experiment should use the prescribed body velocity in
the cell fit itself and test whether that improves the thin-field Γ and M
without sacrificing the broad-field gain. That is an untested hypothesis. It
must retain the point/average distinction and the independent 3D polynomial,
conservation and manufactured checks. A successful candidate would then need
the physical small-domain overlap test, followed by the advancing matched-mesh
force/profile comparison. Increasing panel count or polynomial order alone
has not achieved the required hybrid solution.
