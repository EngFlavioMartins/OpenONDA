# Continuous velocity reconstruction and solenoidal curl in 3D

The new reconstruction is qualified at component level and on the actual
coarse and medium cube meshes. It preserves the specified cell velocity
integrals and produces a locally solenoidal curl, including normal continuity
across tetrahedral faces. Both completed physical comparisons remove the
exterior curl defect, but retain substantial physical errors. Against an
affine source with identical cell moments, near-body reconstruction worsens
in all four mesh/input pairs. The boundary benefit is small on the coarse
mesh and is not consistent on the medium mesh. Advancing force and
velocity-profile agreement remains unachieved.

The subsequent [velocity-projection and mass-flux audit](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/velocity-projection-and-mass-flux-3d.md)
decomposes this correction on both meshes. Projection reduces error relative
to adding the unprojected velocity change in every case. Separately, recovered
reference fluxes show why divergence of reconstructed cell velocity cannot be
interpreted as the FVM solver's mass-conservation error.

This follows the [shared-trace study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/shared-trace-reconstruction-3d.md),
which found that its compact affine vorticity correction changed the induced
curl outside the source support. Global circulation and first-moment budgets
alone did not prevent that effect, and damping those corrections could not
improve either native boundary error.

## Construction and controlled comparison

The [continuous reconstruction](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/continuous_velocity_curl_3d.py)
connects each native face-fan triangle to its actual cell centroid. Velocity
has one value at each shared mesh vertex, one at each face-fan centre, and one
at each cell centre. It is continuous and linear within each tetrahedron.
The subdivision explicitly checks positive volumes, paired interior faces and
the expected physical boundary; nonconforming native edges fail.

The correction's common-node values come from the preceding quadratic velocity
differences. Face-centre weights are minima over adjacent cell weights. Vertex
weights are minima over all incident cells; incident face values are averaged
by face area. This keeps every zero-weight cell untouched. All physical
boundary nodes have zero correction. It changes the previous face-wise
quadratic trace to a continuous piecewise linear one.

The cell-centre node is internal, so changing it does not alter the shared
trace. Its basis function integrates to `V_cell/4`. Choosing its value enforces
the requested cell velocity integral exactly up to floating-point summation.
The point family retains the earlier nonzero cell-integral change. The mean
family retains zero change in every cell; no later circulation or impulse
projection is used.

For this continuous velocity correction `v_h`, define

```text
ω_h = curl v_h.
```

It is constant inside each tetrahedron. Its normal component is continuous
across tetrahedral faces. The zero velocity trace on the physical boundary
makes its zero extension distributionally divergence-free. The reconstructed
velocity `v_h` itself need not be incompressible. If `N=(-Δ)⁻¹` is the free-space
Newton operator, Biot–Savart gives

```text
B(curl v_h) = v_h + grad N(div v_h).
```

The second term is a potential correction. Outside the support, the resulting
velocity therefore has zero curl. The source-panel body correction also has
zero curl away from the body.

For each physical input, the comparison uses two sources with **identical
native-cell circulation and first vorticity moments**:

1. The exact piecewise constant tetrahedral curl, integrated with the qualified
   native-volume Biot–Savart kernel.
2. The previous affine native-cell density reconstructed from those same
   circulation and first-moment values.

The Gaussian complement, exterior particles, native-face-moment starting field,
body mesh and observation positions stay the same. The body response is solved
again for each correction. This isolates information lost beyond the two
retained moments; local solenoidality is part of that information, but the
comparison does not change solenoidality independently of all higher moments.
Comparing either new source with the earlier shared quadratic update also
changes its common-node trace and is not this controlled representation pair.

## Component and physical source checks

The five [component tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_continuous_velocity_curl_3d.py)
pass. They cover affine three-component fields, true polyhedral volume, arbitrary
cell-integral budgets, independent triangle Stokes moments, normal continuity,
compact support, independent tetrahedral volume quadrature, interior and
exterior induced curl, and explicit rejection of nonconforming native edges.
In the manufactured test, the affine source with the same moments has nonzero
exterior curl; the exact tetrahedral curl does not, within differentiation
error.

An initial affine-reproduction assertion detected a `3.11e-15` centre-value
difference. Centre completion now accumulates variations from a local trace
value, reducing cancellation from a constant velocity offset. The original
test tolerance was retained, and all five tests pass with that formulation.

The [physical source builder](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_continuous_curl_sources_3d.py)
replays both preceding quadratic fits, cell integrals and weak curl moments
bitwise before constructing the common-node trace. It uses only the saved
small-mesh velocities and prescribed cube-wall velocity. Exact-zero induction
coefficients are omitted; there is no vorticity threshold or Gaussian core in
the correction.

| Physical source | Small FVM cells | Tetrahedra | Unique tetrahedral faces | Nonzero induction faces |
| --- | ---: | ---: | ---: | ---: |
| Coarse | 2,840 | 71,760 | 146,448 | 110,976 |
| Medium laminar | 16,936 | 416,304 | 843,408 | 561,634 |

| Source and family | Maximum curl normal jump [U∞/D] | Maximum cell-integral difference [U∞D³] | Maximum global first-moment budget difference [U∞D³] |
| --- | ---: | ---: | ---: |
| Coarse, point | 5.21805e-15 | 2.43945e-19 | 1.45717e-16 |
| Coarse, mean | 3.77476e-15 | 3.08320e-19 | 1.38778e-16 |
| Medium laminar, point | 1.35447e-14 | 6.77626e-20 | 5.06539e-16 |
| Medium laminar, mean | 1.11022e-14 | 7.95152e-20 | 2.35922e-16 |

The [independent verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_continuous_curl_3d.py)
checks Stokes' theorem on every tetrahedral face using edge velocity integrals,
without using the reconstruction's inverse-edge gradient or curl routine.
The maximum integrated-flux difference is `1.39e-17`. Independent tetrahedron
sums reproduce the cell moments and global budgets. Its
[source-only verification record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/continuous-curl-source-verification-3d.json)
covers both physical meshes and both families.

These local continuity requirements are consistent with the three-dimensional
`H(curl) → H(div)` sequence and tangential/normal trace distinctions described
by Arnold, Falk and Winther. That mathematical framework motivates the checks;
it does not validate this hybrid solver or imply force agreement.
[Arnold, Falk and Winther, §4.2, author manuscript p. 45](https://sites.math.rutgers.edu/~falk/papers/bulletin-8-13-09.pdf).

## Coarse physical induction

The [paired physical runner](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_continuous_curl_induction_3d.py)
uses the same 7,649 coarse physical targets and 1,728-panel body as the preceding
study, plus 288 derivative evaluation positions. All 864 native coupling faces
and their complete velocity-sampling stencils are included. Twenty-four
deterministic face centres, four on each side, test exterior curl using two
centred-difference steps. Those additional positions lie outside the active
source bounding box.

The coarse calculation completed in about 389 seconds. The new sources give:

| Source | Near-body velocity RMS / U∞ | Native normal-velocity error / U∞ | Native derivative error [U∞/D] | Exterior correction curl RMS [U∞/D] |
| --- | ---: | ---: | ---: | ---: |
| Point native-face-moment baseline | 0.12803324 | 0.00253935 | 0.01754201 | — |
| Point common trace, affine moments | 0.11058357 | 0.00248533 | 0.01734144 | 1.29625e-4 |
| Point common trace, continuous curl | 0.12744378 | 0.00245553 | 0.01728401 | 1.67307e-11 |
| Mean native-face-moment baseline | 0.12801994 | 0.00253971 | 0.01754053 | — |
| Mean common trace, affine moments | 0.11418616 | 0.00258144 | 0.01760436 | 1.28777e-4 |
| Mean common trace, continuous curl | 0.12234229 | 0.00254796 | 0.01755592 | 1.84146e-11 |

The continuous-curl measurements outside the source support are below their
step-halving differences (`3.96e-11` and `2.61e-11`). They are consistent with
zero there. The paired affine sources retain an exterior curl of about
`1.3e-4`, despite identical cell circulation and first moments.

In the controlled point pair, retaining the full curl reduces normal-velocity
error by 1.20% and native derivative error by 0.33%, but increases near-body
error by 15.25%. The corresponding mean pair gives −1.30%, −0.28%, and +7.14%.
Local curl consistency is therefore measurable without being a universal
accuracy improvement.

Relative to its own native-face-moment baseline, the point continuous update
reduces near-body error by only 0.46%, normal-velocity error by 3.30% and native
derivative error by 1.47%. The mean continuous update reduces near-body error
by 4.43%, but leaves normal-velocity and derivative errors 0.325% and 0.088%
above their baseline. These results do not justify promoting the reconstruction
to an advancing transfer.

![Coarse comparison at identical cell moments](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/continuous-curl-coarse-verification-3d.png)

The [combined independent verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/continuous-curl-coarse-verification-3d.json)
checks 59 source records and 178 numerical comparisons, including both physical
source constructions and the coarse induction results. The largest difference
is `1.78e-15`. Native gradients are recomputed by independent accumulation from
the saved induced velocities; their maximum difference is `9.12e-16`. The five
component tests pass, and the plot was visually inspected.

## Medium physical induction

The matched-medium run completed in 6,019 seconds. It evaluates 20,644 induction
positions, including the same 20,356 physical targets and 3,456 coupling faces
as the preceding comparison, with 561,634 nonzero tetrahedral induction faces.
The coarse LES seed and medium laminar seed are distinct physical states, so
their errors cannot be used as a mesh-convergence rate.

| Source | Near-body velocity RMS / U∞ | Native normal-velocity error / U∞ | Native derivative error [U∞/D] | Exterior correction curl RMS [U∞/D] |
| --- | ---: | ---: | ---: | ---: |
| Point native-face-moment baseline | 0.08744658 | 0.00107833 | 0.00351632 | — |
| Point common trace, affine moments | 0.06923677 | 0.00119812 | 0.00360219 | 6.84294e-6 |
| Point common trace, continuous curl | 0.09307893 | 0.00120489 | 0.00360848 | 5.85670e-11 |
| Mean native-face-moment baseline | 0.08744790 | 0.00107824 | 0.00351651 | — |
| Mean common trace, affine moments | 0.07479974 | 0.00108005 | 0.00352322 | 6.23906e-6 |
| Mean common trace, continuous curl | 0.08919413 | 0.00107963 | 0.00352298 | 8.72686e-11 |

In the point pair, retaining continuous curl increases near-body error by
34.44%, normal-velocity error by 0.565% and derivative error by 0.175%.
In the mean pair it increases near-body error by 19.24%, with tiny reductions
of 0.0395% and 0.0069% in the two boundary errors. Both continuous-curl updates
also worsen all three measurements relative to their native-face-moment
starting fields. The exterior-curl measurements are below their step-halving
differences (`6.92e-11` and `1.10e-10`), consistent with zero at those points.

![Coarse and medium comparisons at identical cell moments](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/continuous-curl-verification-3d.png)

The [complete independent verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/continuous-curl-verification-3d.json)
covers 86 source records and 316 checks across both source constructions and
both physical induction results. The maximum numerical difference is
`2.14e-15`; the five component tests pass. The combined plot was inspected.

The table above retains the original normal reference obtained by native
interpolation of full-FVM cell velocity. The subsequent
[accepted-flux comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-accepted-flux-continuous-boundary-medium-laminar/accepted-flux-boundary-3d.json)
also uses the actual conservative reference face flux with identical cell
velocities. Its normal errors are `0.001210915` and `0.001218589` for the point
affine/curl pair, and `0.001071616` and `0.001071805` for the mean pair.
Thus the minute mean-pair normal improvement changes sign under the
conservative-flux observation. None of these four updates has a positive
damping range that improves both accepted normal-flux error and native
derivative error relative to its native-face-moment baseline.

## Remaining physical decision

Correcting local curl structure alone does not resolve the physical
reconstruction error on either mesh. The correction still starts from
a native-face-moment field that differs from the FVM velocity. A subsequent
experiment should distinguish errors in that starting field and in the
reconstructed velocity change, rather than assuming the remaining problem is
an induction or conservation defect. Any eventual transfer must demonstrate
improved advancing forces and velocity profiles in the same small domain.

## Reproduction

Use the OpenONDA Python environment with repository `PYTHONPATH` and a new
output directory for each source or induction run.

```sh
python studies/coupler_accuracy/cube_continuous_curl_sources_3d.py \
  --source studies/coupler_accuracy/results/cube-3d-shared-trace-sources-coarse-qualified \
  --output /private/tmp/cube-continuous-curl-source-coarse
python studies/coupler_accuracy/cube_continuous_curl_induction_3d.py \
  --source /private/tmp/cube-continuous-curl-source-coarse \
  --baseline studies/coupler_accuracy/results/cube-3d-shared-trace-induction-coarse \
  --output /private/tmp/cube-continuous-curl-induction-coarse
```

The source builder requires the five-test XML record and verifies its source
hashes. The physical runner verifies its source construction and preceding
baseline before evaluating a correction. Source archives accompany each run.
