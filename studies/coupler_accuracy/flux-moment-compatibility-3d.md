# Cell velocity and face-flux moment compatibility in 3D

This experiment rejects a proposed shortcut to exact velocity preservation.
Matching each accepted FVM face flux does not make a reconstructed normal
trace compatible with the stored cell velocity integral. Adding ordinary
linear variation across faces helps only modestly. Enforcing the missing
moments by a minimum-change fit reaches a small algebraic residual, but needs
large normal-flow changes in the physical hybrid state. That fit is not a
candidate for advancing transfer.

These are observations of frozen three-dimensional fields. The measured
moment discrepancies and fitted changes are **not hybrid/reference velocity
errors**, and are not additional errors of the pressure solver. The selected
advancing coupling remains in the separate
[long wake comparison](long-wake-comparison-3d.md). The user's force/profile
target remains unachieved.

## The compatibility condition

For a cell origin `c`, the divergence theorem gives

```text
∫cell u dV = ∮boundary (x − c) (u · n) dA
             − ∫cell (x − c) div(u) dV.
```

Writing the face flux and its first moment about face origin `f` as

```text
φ_face = ∫face u · n dA,
m_face = ∫face (x − f) (u · n) dA,
```

a divergence-free volume reconstruction must satisfy

```text
∫cell u dV = Σfaces sign * [(f − c) φ_face + m_face].
```

Zero net cell flux alone does not make the first divergence moment vanish.
Consequently the right-hand side is called the *trace-implied cell integral*
here, rather than an independently reconstructed volume velocity. The study
compares it with both `V_polyhedron * U_cell` and `V_FVM * U_cell`.

This distinction is relevant to a possible reconstruction with continuous
normal velocity. Raviart–Thomas elements explicitly use facet normal moments;
at higher degree they also have interior vector moments. Their normal traces
are continuous across facets. This supplies mathematical context for the
compatibility check, rather than a claim that OpenONDA's stored variables are
already such finite-element degrees of freedom.
[DefElement, Raviart–Thomas definition](https://defelement.org/elements/raviart-thomas.html).

## Actual face geometry, rather than a constant-field false positive

The matched medium meshes have 53,752 full-domain cells and 16,936 small-domain
cells. The same 16,936 native cells are mapped between them. The small box
remains approximately `[-1.5,1.5]^3`, with nominal near-body spacing `0.0625`.

Ignoring `m_face` and using only the summarized face centre and area vector
fails constant-field reproduction on these warped faces: its maximum component
error is `0.0403529823`. Integrating the actual arithmetic-centre face fans
reduces that error to `7.66e-15` over the full mesh and `2.67e-15` over the small
mesh. Area closure divided by volume is at most `1.18e-14`.

The largest local true-fan/stored volume difference is `0.0666531%`, and the
largest centroid displacement is `0.000485580 D`. These differences are kept
explicit; the original FVM mesh summaries are unchanged. The preliminary 4%
constant-field discrepancy was an inadequate diagnostic approximation. It is
not evidence of a corresponding error in the FVM momentum algorithm.

The [new integration operator](native_flux_moments_3d.py) evaluates both normal
flux and its first moment exactly for an affine velocity trace on each actual
triangle. To enforce the stored `φ_face`, it adds a constant scalar normal
trace on each polygon. On a warped face this means a velocity adjustment along
each triangle's own normal. The adjustment preserves the first moment about
the true area centroid, up to floating-point summation.

## Accepted physical snapshots

The [physical runner](cube_flux_moment_compatibility_3d.py) reads canonical full
and hybrid checkpoints from the qualified three-exchange control. The accepted
comparison below is at physical time `0.65`, after 15 FVM steps. Both flows are
laminar, with `ν=0.001` and `dt_FVM=0.01`. The reset initial state at physical
time `0.5` is retained separately in the record and is explicitly labelled as
not an accepted conservative step.

Three traces share exactly the same total face flux:

1. The diagnostic summary approximation with zero within-face moment.
2. A constant shared velocity on each face, integrated on its actual triangles.
3. A shared affine velocity trace, obtained by neighbouring-cell least squares
   and linear extrapolation. Boundary faces retain their stored constant
   velocity traces. Each solver uses only its own cell and boundary data.

All entries below are volume-weighted RMS of the **trace-implied mean minus
stored cell velocity**, divided by `U∞`. Near-body means
`max(|x_cell|,|y_cell|,|z_cell|) < 0.8`.

| State and trace | All shared cells | Near-body cells |
| --- | ---: | ---: |
| Full FVM, summary approximation | 0.02014838 | 0.06398225 |
| Full FVM, actual constant trace | 0.01996239 | 0.06333129 |
| Full FVM, actual affine trace | 0.01899390 | 0.06112008 |
| Hybrid FVM, summary approximation | 0.02028151 | 0.06399010 |
| Hybrid FVM, actual constant trace | 0.02009664 | 0.06333881 |
| Hybrid FVM, actual affine trace | 0.01912489 | 0.06112397 |

The accepted flux divergence RMS on the shared cells is `1.17e-15 U∞/D` for
the full FVM and `2.05e-15 U∞/D` for the hybrid. Thus mass conservation and
this first-moment discrepancy coexist in both discrete solutions. Correcting
warped-face integration and adding ordinary affine variation do not remove
the physical near-body discrepancy.

The [independent verifier](verify_flux_moment_compatibility_3d.py) uses a
three-point triangle quadrature instead of the analytic moment formula. It
reintegrates both actual traces in all four snapshots, checks their conservative
fluxes and cell sums, and recomputes 80 regional metrics. The largest metric
difference is `1.39e-17`. All 799 original frozen source/input files were checked
unchanged before and after the physical run.

The [complete moment record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/native-flux-moment-compatibility-medium/flux-moment-compatibility-3d.json)
and [independent verification](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/native-flux-moment-verification-medium.json)
retain the source hashes, face traces, moments, mapped cells and regional data.

## What a hard moment fit would cost

The [normal-moment lift](native_normal_moment_lift_3d.py) adds a scalar normal
velocity of the form

```text
δu_n(x) = a_face · (x − area_centroid_face).
```

It has zero total flux on every face. The cube wall remains fixed. Interior
faces use one shared correction, while within-face variation is allowed on
the other outer faces. This latter freedom is a diagnostic assumption, not a
new physical boundary condition. The fit minimizes integrated squared normal
change over all free faces, subject to the cell first-moment constraints.

The surface covariance supplies energy-normalized modes. A two-mode control
uses its dominant directions. A three-mode calculation includes all resolved
directions, retaining eigenvalues larger than sixteen machine epsilons times
the largest face eigenvalue. The solver checks both primal constraints and a
dual energy bound; its minimum claim concerns this stated numerical space.

For the hybrid state at physical time `0.65`, preserving stored cell velocity
gives:

| Quantity | Two face modes | All resolved face modes |
| --- | ---: | ---: |
| Velocity-moment constraint RMS / U∞ | `6.89e-11` | `4.61e-11` |
| Normal change RMS on all free faces / U∞ | 0.374438 | 0.252615 |
| Normal change RMS on coupling outer faces / U∞ | 0.060125 | 0.060089 |
| Maximum normal change anywhere / U∞ | 38.6939 | 46.8227 |
| Maximum normal change on outer faces / U∞ | 0.549990 | 0.548720 |
| Absolute relative primal/dual energy difference | `3.96e-12` | `6.64e-13` |

Using the stored-volume momentum instead changes the all-resolved-mode RMS
from `0.252615` to `0.252605`; that geometry convention does not remove the
large fitted variation. Direct triangle quadrature independently checks the
zero face-flux increments, first moments, cell changes and surface energy.
Across these four fits its maximum cell-velocity difference is `3.91e-13` and
relative energy difference is `9.13e-15`. The source and arrays are in the
[hybrid lift record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/native-normal-moment-lift-medium-hybrid/normal-moment-lift-3d.json).

The full-domain control also completes all four fits. Its all-resolved-mode
maximum normal change is `46.7971 U∞`, at the same native face near a cube
corner as the hybrid maximum. A
[mapped-face comparison](compare_normal_moment_lifts_3d.py) checks matching
centroids and areas on all 54,408 small-mesh faces before comparing the fits:

| Identical face region, stored-velocity target | Full FVM fit RMS / U∞ | Hybrid fit RMS / U∞ |
| --- | ---: | ---: |
| All 52,464 free shared faces | 0.252188 | 0.252615 |
| 14,880 near-body free faces | 0.707654 | 0.707407 |
| 3,456 coupling faces | 0.052680 | 0.060089 |

The fits are posed on their respective whole domains; only these reported
norms use the same faces. The full-domain counterfactual also frees variation
on its outer physical boundary. Its much larger total face area would dilute
an unmatched whole-domain RMS, which is why the mapped norms matter here.
The comparable near-body changes show that this particular hard-fit problem
is not unique to coupling. The
[full lift record](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/native-normal-moment-lift-medium-full/normal-moment-lift-3d.json)
and [matched comparison](/Users/flaviomartins/OpenONDA/studies/coupler_accuracy/results/interface-precision-frozen-workspace-qualified/studies/coupler_accuracy/results/native-normal-moment-lift-comparison-medium.json)
retain the complete eight-fit evidence and common-face metrics.

There is also a simple obstruction if **all boundary first moments are fixed**.
Internal face changes cancel in the global integral. In this hybrid snapshot
the required global cell-integral correction is nonzero, giving a
volume-weighted velocity-discrepancy lower bound of `0.00130733 U∞`, regardless
of how the interior normal traces are changed. This is a bound for these
fixed traces and imposed cell-mean constraints. It is not a lower bound on
all possible hybrid/reference force or profile errors.

Five component tests cover affine 3D divergence-theorem identities, independent
quadrature on warped faces, shared-face cancellation, unchanged wall and total
face flux, a feasible moment lift with a dual energy check, and the obstruction
with fixed boundary moments. Both test files pass in the frozen workspace.

## Consequence for transfer design

The hard fit is rejected as a transfer candidate. Its small residual conceals
large normal-flow oscillations, including changes at the coupling boundary.
No production geometry, boundary condition, particle field or time-advancing
solver is changed by this study.

A compatible reconstruction must explicitly decide how stored cell velocities
and conservative face fluxes are interpreted. Enforcing the selected cell
means at the cost of large unresolved face variation is not justified by
the user's desire for close profile agreement. The next reconstruction needs
to control both the induced velocity and its boundary derivatives, and must
be tested through the advancing hybrid/reference comparison. Matching these
necessary moments alone is not an acceptance criterion.
