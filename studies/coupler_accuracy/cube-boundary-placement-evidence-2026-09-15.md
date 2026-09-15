# Cube wake: established mechanism and limits of a boundary-placement claim

This is a synthesis of completed experiments, checked against their raw records
on 15 September 2026. It adds no simulation or production change. The underlying
studies were committed in `ca8d48e3` and `212a8fc3`.

The evidence establishes a finite-resolution inconsistency between particle
vorticity, its induced velocity, transfer, and evolution. It identifies a major
source of the measured downstream disturbance. It does not yet establish the
fraction of the complete reference error caused by that mechanism, or a safe
distance between the cube and the coupling boundary.

## What the formulation fails to preserve

Let `w` be the reconstructed Gaussian particle vorticity, `u` the induced
velocity, and `q=curl(u)`. In the free-space Helmholtz decomposition,

```text
w = q + ell,     ell = grad(psi).
```

The extra component `ell` produces no Biot–Savart velocity. Nevertheless, the
tested transfer and transport can turn it into velocity. Thus two particle
representations of the same physical velocity need not remain equivalent.

Assigning cell-integrated strengths does not itself enforce this compatibility:
`div(sum_p Gamma_p zeta_p) = sum_p Gamma_p . grad(zeta_p)` need not vanish.
Spatial weights, finite Gaussian reconstruction, and evolution must also be
compatible. The separate contributions of every operation to the accumulated
`ell` have not been measured from startup.

Two failures have independent controlled reproductions:

1. **Transfer can change a velocity field that already matches its donor.**
   For a physically matching FVM donor, the represented-field target is
   `w_new = w - eta*ell`. Its change has curl
   `-grad(eta) cross ell`, generally nonzero for a spatial authority weight.
   The actual coefficient blend and Gaussian correction add their own
   representation effects. See the [transfer derivation and cube control](cube-transfer-cause-2026-09-15.md).
2. **Transport can make velocity depend on the invisible component.**
   With `J_ij=partial_j u_i`, the tested positive-transpose extension gives
   `-(u.grad)ell + J.T ell = -grad(u.ell) + 2 J.T ell`.
   Biot–Savart removes the first term but generally retains the second.
   Finite-core particle transport has an additional remainder, explicitly
   retained in the actual cube budget. See the [derivation and assumptions](longitudinal-transport-analysis-2026-09-15.md).

This identifies the missing compatibility property. It is not a proposal to
reverse physical vortex stretching or to assert that VPM cannot represent
recirculation.

## Evidence with controlled alternatives

All controls are three dimensional. Velocity RMS uses the freestream as its
scale, never a near-zero local reference velocity.

| Experiment | Measurement | What it establishes |
| --- | --- | --- |
| Transfer of a zero-velocity state containing a gradient vorticity component | Initial velocity RMS `4.44e-15 U`; after the cube's spatial blend, `6.38e-3 U`; constant-weight control, `2.22e-15 U` | Spatial transfer creates velocity despite physically matching donor and recipient. |
| The same gradient component in an exact incompressible affine strain | Spurious velocity-rate RMS `1.02e-2 U^2/D`; gradient-preserving control, `5.42e-14 U^2/D` | Evolution activates the component without a body, recirculation, FVM boundary, LES, tree approximation, or time integrator. |
| Actual saved cube state at reduced time 6 | `RMS(w-curl(u))/RMS(w) = 21.4%` on the 196 downstream 3D probes (`1.25 <= x/D <= 1.62`) | The problematic component is present in the simulation, rather than only in a manufactured input. |
| Differentiated particle state at reduced time 6 | Longitudinal contribution `+0.00627055 U^3/D`; actual local asymmetry-energy derivative `+0.00986705 U^3/D` | Approximately 64% of this instantaneous local growth comes from the identified term. This is not 64% of accumulated velocity or force error. |
| Conservative partial correction, prescribed-reference replay from 6 to 7 | Seam velocity error `0.12968 -> 0.11965 U`; outer-wake error `0.10259 -> 0.08944 U` | A targeted intervention improves the measured reference agreement. Improvement persists at half the time step. The residual remains substantial. |

Raw records: [transfer](results/cube-wake-cause-2026-09-15/transfer-nullspace/probe.json),
[manufactured transport](results/cube-wake-drift-2026-09-15/nullspace-manufactured/probe.json),
[vorticity consistency and native operators](results/cube-wake-drift-2026-09-15/operator-audit/audit.json),
[instantaneous budget](results/cube-wake-cause-2026-09-15/represented-evolution-budget/budget.json),
[correction comparison](results/cube-compact-flux-evolution-2026-09-15/comparison.json).

In the manufactured controls, halving integration spacing from `0.06 D` to
`0.03 D` leaves the spurious transfer velocity and transport rate essentially
unchanged while the initial velocity and control approach roundoff. The core
radius stays `0.066 D`. This is quadrature refinement, not a mesh/core convergence
study of the full solver. The cube budget also checks diagnostic quadrature and
periodic-box size, with the native free-space/body remainder kept explicitly.
Those diagnostic box changes do not move the FVM boundary.

The replay uses the saved centred-lattice cube configuration without alignment,
the same initial particle state, and prescribed fine-reference donors. No FVM
solve or boundary feedback occurs. Donor states between the saved endpoints at
6 and 7 are linearly interpolated; the final comparison uses the native time-7
reference. The correction is partial and remains expensive; its source impulse
is sensitive to time-step refinement. These experiments do not qualify forces,
a corrected clean-start trajectory, or the present production tutorial. See
the [complete qualification and limits](cube-compact-flux-evolution-2026-09-15.md).

## What the evidence says about the boundary

The cube occupies `[-0.5,0.5]^3 D`. The tested FVM outer faces lie at
`+/-1.5 D`, one diameter from the corresponding body faces. The transfer box
ends at `+/-1.25 D`. These are different surfaces and must be distinguished
when changing the geometry.

At reduced time 8, at approximately `(1.44,-0.18,0) D`, VPM gives
`u_z=0.35793 U`, versus reference `0.0000831 U`. Particles with
`1.25 <= x_p/D <= 1.65` contribute `0.32355 U` at that point; the near-body
population contributes only `0.00151 U`. This locates the sources of the
existing disturbance, without by itself establishing where they acquired it.
[Source decomposition](results/cube-wake-drift-2026-09-15/source-localization/localization.json).

The reference centreline reattachment reaches approximately `x/D=1.48` at
time 3 and `1.72` at time 5, beyond the FVM downstream face. This supports an
association with handing part of the developing recirculation region to VPM.
It does not prove that recirculation or boundary proximity is necessary for
the defect: the manufactured forward-flow control contains neither.
[Native comparison and reattachment definition](cube-wake-drift-2026-09-15.md).

The implemented mixed condition prescribes normal velocity and the normal
derivative of tangential velocity. Tangential velocity is reconstructed using
the FVM interior. At time 8, the corrected-lattice run has a maximum FVM–VPM
full-velocity mismatch of `0.588 U` at the outflow, despite a maximum normal
mismatch of `5.46e-16 U` and converged interface iterations. This explains why
the convergence flag does not certify full-velocity agreement; it does not
prove the mixed boundary condition is mathematically invalid or justify adding
another velocity constraint. [Raw boundary traces](results/cube-wake-drift-2026-09-15/advancing-comparison/comparison.json),
[boundary reconstruction](../../source/solvers/fvm/fields/mixed_velocity_boundary.py).

No completed matched boundary-distance sweep was found in the investigation's
reports. Earlier cropped-FVM boundary experiments use different short diagnostic
problems and do not isolate distance in this developed fine-grid wake. Therefore
neither a universal minimum distance nor the effectiveness of moving this
boundary is established. The demonstrated mechanism depends on `ell`, velocity
gradients, and spatial transfer gradients; distance from the body is only an
indirect possible control on these quantities.

## Wording supported for a thesis or paper

> In the tested compact-domain cube configuration, the particle representation
> developed a discrepancy between its reconstructed vorticity and the curl of
> its induced velocity. Controlled three-dimensional experiments demonstrated
> that spatial vorticity blending and the tested stretching formulation can
> convert the velocity-invisible component into spurious velocity. An
> instantaneous budget identified this mechanism as a major contributor to
> downstream disturbance growth, and a targeted partial correction improved
> short-interval reference agreement. These results identify a consistency
> limitation of the tested formulation. Although the observed error is located
> near the downstream transfer region, a minimum permissible boundary distance
> has not been established.

A stronger placement claim requires a matched test that moves the downstream
FVM face and transfer region beyond the developing reference recirculation,
while preserving shared near-body cells, particle spacing/core, overlap width,
time policy, fluid models, and FVM schemes. A reference-fed cropped-FVM control
at both positions separates boundary/discrete-operator error from VPM and
transfer error. Both field agreement and the identified source term must
improve; a shifted error maximum or a healthier run alone would not suffice.
