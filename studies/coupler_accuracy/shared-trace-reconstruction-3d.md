# Conservative shared-face reconstruction in the fully 3D cube

The shared-face correction passes its circulation and first-moment budgets on
both physical meshes. Conservation alone does not make it the most accurate
source representation. On the coarse physical cube, its cell-average variant
reduces near-body velocity error by 9.78% relative to its native-face-moment
baseline, but increases native normal-velocity error by 9.75% and native
tangential-derivative error by 2.11%. It remains an isolated reconstruction
experiment. Advancing force/profile agreement is unachieved.

The medium laminar comparison is now complete and independently verified.
Its mean shared update reduces near-body reconstruction error by 11.32%, but
increases native normal-velocity error by 4.34% and native derivative error by
2.60%, relative to its mean native-face-moment baseline. On both physical
snapshots, no positive damping factor for either shared-update direction
reduces either native boundary error. A separate exterior-curl measurement
confirms that the compact raw correction changes the induced velocity's curl
outside its support. This identifies a representation limitation, not a
demonstrated explanation or cure for the advancing drag error.

This follows the [native boundary sampling study](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/sampled-native-face-followup-3d.md).
The small FVM box remains approximately `[-1.5,1.5]^3` around the unit cube.
Every vector component and spatial direction is active. The coarse snapshot
uses equilibrium Smagorinsky, while the medium snapshot uses the laminar seed
from the latest live comparisons. These are two physical source tests, **not
a coarse-to-medium convergence study**.

## What is conserved

Let `Q_c` be the cell velocity integral, `Γ_c` its vorticity integral, and
`M_c,ij = ∫cell (x-c)_i ω_j dV` its first central vorticity moment. The native
triangle geometry gives the weak curl identities

```text
Γ_c    = ∮cell n × u dA
M_c,ij = ∮cell (x-c)_i (n × u)_j dA − (e_i × Q_c)_j.
```

The [new update](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/native_shared_trace_update_3d.py)
applies the overlap weight to the change of velocity trace before taking this
curl. An interior face receives `w_f = min(w_owner,w_neighbour)`, and its
velocity, gradient and Hessian changes all receive that same scalar weight.
The cell-integral change receives `w_c`. Face weights are constant on each
face polynomial; this is a discrete compact-support rule, not exact integration
of a continuous taper.

All weighted trace changes vanish on the physical boundary. The same interior
face contributes with opposite signs to its two cells, giving

```text
Σ ΔΓ_c = 0
ΔH_ij = Σ [ΔM_c,ij + c_i ΔΓ_c,j] = −(e_i × Σ ΔQ_c)_j
ΔI = ½ ∫ x × Δω dV = Σ ΔQ_c.
```

The cell-average reconstruction retains `Q_c = V_poly,c U_c` exactly, so
`ΔQ_c=0` and the raw-source impulse increment vanishes. A point-value
interpretation instead changes the reconstructed cell integral; its impulse
changes by exactly the stated integral budget. Neither variant uses a later
circulation or impulse projection.

The resulting sources are

```text
Γ_source,c = w_c Γ_native,c + ΔΓ_c
M_source,c = w_c M_native-face,c + ΔM_c.
```

The conservative update therefore starts from the native-face-moment
representation. It does **not** preserve the impulse of the zero-moment P0
control. On the coarse snapshot, adding those baseline cell-average moments
changes streamwise raw impulse by `+0.1055898894 U∞D³`; on the medium laminar
snapshot the change is `+0.05348710890 U∞D³`. The shared correction then
preserves each of those baselines. A complete advancing emission procedure
must account for the starting representation and physical impulse evolution.

A separate [endpoint check](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_native_moment_impulse_3d.py)
uses the medium laminar full-FVM snapshots at physical times 0.5 and 1.5.
The streamwise impulse contributed by those baseline moments increases from
`0.05348710890` to `0.06390408670 U∞D³`. Its mean difference rate is therefore
`0.01041697780 U∞²D²`. This establishes that the representation change is not
a constant initial offset. Independent global boundary sums reproduce the
cell-moment sums within `1.01e-13`, and both native circulation fields replay
within `1.80e-14` after division by cell volume.

That difference rate is **not a drag prediction**. The diagnostic uses two
full-FVM states, has no actual particle-renewal history, and does not include
the terms needed to infer force from a finite control volume. Circulation,
Gaussian complements and exterior-particle contributions cancel in the
difference between these two representations, so neither the exterior cutoff
nor Gaussian core explains this particular measurement. Its numerical record
is [native-moment-impulse-3d.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/cube-3d-native-moment-impulse-medium-laminar/native-moment-impulse-3d.json).
The [combined coarse and endpoint verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-coarse-and-impulse-verification.json)
checks 65 source records, 309 numerical comparisons and 62 independent array
comparisons. Its largest array difference is the `1.01e-13` global moment
summation difference above; the native gradient checks remain within
`1.30e-15`.

The quantities above describe raw affine source vorticity. Biot–Savart induces
a solenoidal velocity; its curl need not equal a nonsolenoidal input vorticity
field. These budgets are not claims about local cell moments of that induced
velocity's curl.

## Native geometry and velocity interpretation

The [source builder](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_shared_trace_sources_3d.py)
uses only small-mesh velocities and the prescribed zero cube velocity for its
quadratic reconstruction. It retains the original Gaussian complement and
exterior particles. Zero-weight cells receive no update, and outer boundary
placeholders cannot affect an active source face.

Two interpretations of the same FVM values are compared: point values at native
FVM centres, and averages over the actual triangle-fan polyhedra. Neither is
asserted to be the exact physical interpretation of the numerical FVM state.

The FVM's scalar cell volumes and the source polyhedron volumes differ locally:
the maximum relative differences are 0.06719% on the coarse mesh and 0.06661%
on the medium mesh. The conservative mean reconstruction preserves `U_c` and
therefore the transported FVM state `V_FVM,c U_c`. Its source integral uses
`V_poly,c U_c`. These are explicitly distinct measures. Scaling velocity by
`V_FVM/V_poly` would create variation in an otherwise uniform velocity field
and is not used.

## Independent source budgets

The four [component tests](/Users/flaviomartins/OpenONDA/tests/coupler/test_native_shared_trace_update_3d.py)
use a warped 5³ mesh and multiple three-component polynomial states. They
compare against independent triangle quadrature, check arbitrary cell-integral
changes, retain zero-weight cells exactly, and reject noncompact boundary
updates. All four pass.

Independent sums from the saved physical sources give the following values.
Circulation norms have units `U∞D²`; impulse and first-moment errors have units
`U∞D³`:

| Source snapshot and mean update | Norm of total ΔΓ | Streamwise ΔI | Maximum first-moment budget error |
| --- | ---: | ---: | ---: |
| Coarse, cell-weighted completed curl | 5.38603e-5 | −0.0566966451 | 0.0567020533 |
| Coarse, shared weighted trace | 1.93459e-16 | −1.34224e-16 | 4.56015e-16 |
| Medium laminar, cell-weighted completed curl | 1.66661e-6 | −0.00145272202 | 0.00145296086 |
| Medium laminar, shared weighted trace | 2.32674e-16 | 2.75062e-16 | 2.81188e-16 |

This establishes the difference between weighting completed cell curls and
curling a shared weighted trace. It does not establish that the conserved
baseline is the most accurate physical field.

## Coarse physical induction and body response

The [physical runner](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/cube_shared_trace_induction_3d.py)
holds the body at 1,728 panels and evaluates its exact f64 triangular kernel.
The constant-volume-plus-Gaussian baseline is evaluated first, followed by the
affine circulation/moment correction. The body response is solved as a linear
correction to the baseline, with zero net source flux enforced by the existing
Neumann solver.

The coarse calculation evaluates 7,649 distinct positions: independent cell
and wall targets, body collocation points, all 864 coupling-face centres, and
the complete native gradient stencils requiring 3,088 cell samples. Every
stencil velocity comes from the reconstructed sources and solved body. Full
FVM values are comparison data only.

All normal-velocity observations now use true native unit normals and vector
area magnitudes, matching the current live convention. Earlier frozen reports
used a slightly different area convention. Native tangential derivatives keep
their existing convention and can be compared directly.

| Source | Near-body velocity RMS / U∞ | Native normal-velocity error / U∞ | Native derivative error [U∞/D] |
| --- | ---: | ---: | ---: |
| Constant volume P0 | 0.17043235 | 0.00367410 | 0.01781649 |
| Linear-face moments | 0.12235849 | 0.00270870 | 0.01830906 |
| Point native-face moments | 0.12803324 | 0.00253935 | 0.01754201 |
| Point quadratic, completed cell curls weighted | 0.10681367 | 0.00252488 | 0.01766567 |
| Point quadratic, shared trace update | 0.10772221 | 0.00262849 | 0.01763533 |
| Mean native-face moments | 0.12801994 | 0.00253971 | 0.01754053 |
| Mean quadratic, completed cell curls weighted | 0.11305673 | 0.00247735 | 0.01768604 |
| Mean quadratic, shared trace update | 0.11549719 | 0.00278745 | 0.01791067 |

Relative to its mean native-face baseline, the conservative update lowers
near-body error by 9.78% and independent wall penetration by 5.66%, while
worsening both boundary measurements. Relative to P0, near-body error falls
32.23% and normal-velocity error falls 24.13%, but derivative error rises 0.53%
and wall penetration rises 15.41%. The reference chosen for a percentage
materially changes the conclusion.

The point shared update also trades near-body improvement for slightly worse
boundary measurements relative to its own baseline. It conserves total
circulation but has the nonzero impulse change required by its changed cell
integrals. Neither point nor mean results justify claiming that conservation
has solved the coupling error.

![Coarse shared-trace source comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-coarse-verification.png)

The P0, linear-moment and point cell-weighted quadratic fields reproduce their
previous qualified versions within `1.40e-13`; the largest difference occurs
at coupling-face positions. Their native derivatives reproduce within
`2.67e-15`. The calculation completed in about 208 seconds.

The [independent verifier](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/verify_shared_trace_induction_3d.py)
checks 55 source records and 304 scalar/array-valued numerical comparisons for
the coarse experiment. It independently rebuilds the Gauss gradients and face
derivatives from stored velocities. The maximum metric difference is
`5.78e-15`; 60 independent array comparisons differ by at most `1.30e-15`.
The plot was visually inspected. The numerical record is
[shared-trace-coarse-verification.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-coarse-verification.json).

## Medium laminar physical induction

The medium run uses 16,936 small-FVM cells and the matched 53,752-cell full
reference at physical time 0.5, with spacing 0.0625. It evaluates 20,356
distinct positions, including all 3,456 coupling faces, 13,024 native stencil
cell samples, 1,536 independent wall samples, and the same 1,728 body panels.
Each cell-velocity region contains 256 deterministic samples. Direct induction
and all body responses completed in about 2,963 seconds.

| Source | Near-body velocity RMS / U∞ | Native normal-velocity error / U∞ | Native derivative error [U∞/D] |
| --- | ---: | ---: | ---: |
| Constant volume P0 | 0.13237144 | 0.00208091 | 0.00455366 |
| Linear-face moments | 0.08400854 | 0.00163293 | 0.00433130 |
| Point native-face moments | 0.08744658 | 0.00107833 | 0.00351632 |
| Point quadratic, completed cell curls weighted | 0.06823475 | 0.00115292 | 0.00366784 |
| Point quadratic, shared trace update | 0.06810064 | 0.00117544 | 0.00362554 |
| Mean native-face moments | 0.08744790 | 0.00107824 | 0.00351651 |
| Mean quadratic, completed cell curls weighted | 0.07769106 | 0.00108833 | 0.00364629 |
| Mean quadratic, shared trace update | 0.07755182 | 0.00112503 | 0.00360778 |

The mean shared update improves near-body error by 11.32% and wall penetration
by 4.33% relative to its mean native-face-moment baseline, but worsens native
normal velocity by 4.34% and its derivative by 2.60%. The point shared update
improves near-body error by 22.12% while worsening those two boundary errors by
9.01% and 3.11%. Relative to P0, the mean shared source reduces all three
tabulated errors, but increases independent wall penetration by 38.47%.
These are frozen reconstructed-field comparisons, not changes to an advancing
FVM solution or measured drag.

![Both physical shared-trace comparisons](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-induction-verification.png)

The [combined verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-induction-verification.json)
checks both physical runs, both source constructions, the four component tests,
and the endpoint impulse audit. It checks 115 source records, 613 numerical
comparisons and 122 independent array comparisons. The largest metric
difference is `6.67e-15`; the largest array difference remains the endpoint
global moment summation difference of `1.01e-13`. The combined plot was visually
inspected. Coarse and medium rows use different physical seeds; their values
must not be read as a mesh-convergence rate.

## Damping cannot rescue these update directions

With sources and body geometry frozen, induction, the constrained body solve
and the boundary mass correction are linear. For the baseline error `e` and
the full update `δu`, a scale `λ` gives the exact identity

```text
E(λ)² = <e,e> + 2 λ <e,δu> + λ² <δu,δu>.
```

The [response diagnostic](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/shared_trace_response_3d.py)
replays both endpoints and checks this identity at five scales. In both point
and mean families on both snapshots, `<e,δu>` is positive for native normal
velocity and native tangential derivative. Consequently every positive scale
increases both errors. Underrelaxation of this fixed direction cannot make
either native boundary measurement more accurate than its starting field.
This conclusion is specific to these reconstructions, baselines and frozen
states; it does not exclude a different conservative update. Reference errors
are used only for diagnosis, not to select a production damping factor.
The numerical record is [shared-trace-response-3d.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-response-3d.json).

## Compact raw sources do not imply compact physical curl

The [curl diagnostic](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/shared_trace_curl_3d.py)
evaluates only the shared correction minus its corresponding native-face-moment
baseline. Their Gaussian complements cancel exactly, and the body source
potential has zero continuous curl outside the body. Thus this isolates the
physical curl change induced by the raw affine correction.

All active source cells lie within approximately `[-1.25,1.25]^3`. The test
selects four deterministic coupling-face centres on each of the six sides.
All 288 velocity evaluation positions, used for three Cartesian derivatives
at two centred-difference step sizes, are at least 0.24969D outside the active
source bounding box. The raw vorticity correction is exactly zero there.

| Snapshot and correction | Exterior curl RMS [U∞/D] | RMS change on halving derivative step [U∞/D] |
| --- | ---: | ---: |
| Coarse, point | 7.08603e-4 | 3.40627e-9 |
| Coarse, mean | 7.47551e-4 | 3.93393e-9 |
| Medium laminar, point | 2.05758e-4 | 4.19165e-8 |
| Medium laminar, mean | 2.17368e-4 | 4.83550e-8 |

The induced velocity's exterior curl is nonzero well above the differentiation
noise. The raw piecewise affine fields have both nonzero interior divergence
and normal jumps between cells. An independent divergence-theorem sum of their
volume and triangle-face terms agrees within `1.77e-15`. Conservation of the
global moments has therefore not produced a locally divergence-free raw curl
correction; Biot–Savart projects it nonlocally.

These are 24-point diagnostic norms, not whole-boundary norms. They must not be
equated with the native tangential-derivative errors or used to assign a share
of the drag discrepancy. The result establishes a limitation of the current
affine correction and motivates a locally solenoidal reconstruction test; it
does not establish that this limitation dominates the production error.
The saved fields and measurements are recorded in
[shared-trace-curl-3d.json](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/shared-trace-curl-3d/shared-trace-curl-3d.json).

## Consequence for the next coupling decision

The current experiment tests whether correcting the overlap's conservation
defect also improves the actual field. Both physical snapshots give a tradeoff,
so this update must not be promoted as an accuracy fix. Global circulation and
impulse identities are necessary checks but do not constrain the local
solenoidal projection. A representation that preserves local curl structure
now warrants a separate component experiment. Particle emission and force
histories still require their own validation.

The literature also supports testing the near-wall representation separately
from emitted particles. Stock, Gharakhani and Stone use a fully 3D hybrid
method that excludes the strongest near-wall vorticity from particle reset
and supplies its influence through a boundary-element solution. Their overlap
algorithm retains buffers at both the wall and the outer Eulerian boundary.
That is a distinct route to investigate if richer volume reconstruction remains
insufficient; it is not a validated change to this solver.
[Stock et al., AIAA 2010-4553, §II.C](https://markjstock.org/research/AIAA-2010-4553.pdf).

Valentin et al. likewise use a distinct representation of the inner Eulerian
region in their 3D sphere and wing studies, together with particle generation
in an overlap. They report residual discrepancies and identify particle
initialization and boundary representation as remaining issues. This supports
separating those components, not a claim of near-roundoff agreement.
[Valentin et al., AIAA 2024-3865, author manuscript](https://www.researchgate.net/publication/383260525_Hybrid_Eulerian-Lagrangian_Method_for_Complex_3D_Viscous_Flows).

## Reproduction

Use the OpenONDA Python environment, repository `PYTHONPATH`, single-threaded
BLAS and new output directories. The source construction must precede its
physical induction run:

```sh
python studies/coupler_accuracy/cube_shared_trace_sources_3d.py \
  --oracle studies/coupler_accuracy/results/cube-3d-oracle \
  --coarse-replay studies/coupler_accuracy/results/cube-3d-boundary-quadratic-overlap-1728-qualified \
  --output /private/tmp/cube-shared-source-coarse
python studies/coupler_accuracy/cube_shared_trace_induction_3d.py \
  --source /private/tmp/cube-shared-source-coarse \
  --oracle studies/coupler_accuracy/results/cube-3d-oracle \
  --coarse-replay studies/coupler_accuracy/results/cube-3d-boundary-quadratic-overlap-1728-qualified \
  --output /private/tmp/cube-shared-induction-coarse
```

For the medium source and induction, use
`cube-3d-medium-laminar-oracle` and omit `--coarse-replay`. The fixed panel
geometry defaults to the previously qualified 1,728-panel cube. An explicit
`--induction-checkpoint` can reuse a completed source evaluation only when its
source hash, source names and complete target array match exactly.

The first source-builder attempts stopped at a state-axis mismatch before
reconstruction. Those directories lack a completed source report; only the
`-qualified` source directories are used here. The runner's input axis was
corrected, and all three coarse source controls then replayed bitwise.
