# Optional infinite-span cylinder model

The cylinder reference solves an extruded one-diameter section with slip span
boundaries. A spanwise-invariant solution represents an infinite cylinder, not
a finite one-diameter vortex wake. `PlanarInduction` implements that model
without a panel solve. It is opt-in; the ordinary three-dimensional cube
operators and cylinder tutorial defaults are unchanged.

The choice is appropriate to the intended Re=150, infinite-cylinder benchmark:
Barkley and Henderson's primary stability analysis finds the unconfined
two-dimensional cylinder wake first becomes linearly unstable to 3D
perturbations at Re=188.5±1.0
([Journal of Fluid Mechanics, 1996](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/threedimensional-floquet-stability-analysis-of-the-wake-of-a-circular-cylinder/61575FBF0BC45054592D46382DEF30BB)).
This supports, but does not itself validate, the model choice here. Finite ends,
different forcing/confinement, and higher Reynolds numbers require separate
physical qualification; the reference span controls and live invariance checks
remain necessary.

## Numerical contract

Set `induction=vpm.PlanarInduction(span=1.0, plane_z=0.0)`. The VPM stores one
particle per XY node at `z=plane_z`. Its stored strength is
`Gamma_z = omega_z * h**2 * span`, its volume is `h**2 * span`, and its
circulation is `Gamma_z / span`. Other strength components are prohibited.
The represented Gaussian is

\[
\omega_z(x,y)=\sum_p\frac{\Gamma_{z,p}}{\pi L\sigma_p^2}
\exp[-r_p^2/\sigma_p^2],\qquad
\mathbf u_p=\frac{\Gamma_{z,p}}{2\pi L}
\frac{1-e^{-r_p^2/\sigma_p^2}}{r_p^2}(-\Delta y,\Delta x,0).
\]

Here `sigma` is the existing OpenONDA core-radius convention, not the normal
distribution standard deviation. The velocity Jacobian is differentiated
analytically, including its finite source-centre curl. All targets are
independent of z. The stage strength derivative is identically zero: there is
no vortex stretching in this model. Evaluation currently uses exact direct
pair summation on the selected Taichi device, so its complexity is O(N²).

Selecting the backend also selects the following compatible operators:

* **GBD:** complete XY M4-prime remeshing; a five-point molecular Laplacian;
  substeps with `nu*dt_sub/h**2 <= 0.12`, below the non-sign-reversing limit
  1/8; no z diffusion. Pruning recovers signed circulation, both first moments,
  and all three second moments. Its correction and residual diagnostics feed
  the existing coupling acceptance gate. Capacity exhaustion raises an error
  rather than silently reducing the resolved wake.
* **Renewal:** one XY lattice, no authority taper at artificial z ends,
  represented Gaussian convolution in two dimensions, and an XY velocity
  circulation trace multiplied by the represented span. The same fixed
  lattice phase is used for renewal and GBD.
* **Solid exclusion:** native FVM wall triangles provide the planar GBD mask
  and renewal classifier. No panel body is needed to retain the solid mask.
* **FVM consistency:** every transfer checks all extruded XY cell stacks,
  before discarding transverse vorticity. Maximum spanwise velocity and
  variation must remain below `spanwise_tolerance` (default 0.001) times the
  larger of freestream and cell-stack mean velocity scales. Failure stops the
  run rather than silently accepting a three-dimensional donor field.
* **Diagnostics:** velocity, curl, Jacobian, strain, line/surface samplers,
  particle vorticity, enstrophy, and helicity use the planar model. Gaussian
  pair integrals give energy over the infinite XY plane times the represented
  span. Unbounded energy is infinite when net circulation is nonzero; this is
  labelled `planar_unbounded_energy_diverges`. Net circulation below 1e-7 of
  circulation L1 is treated as roundoff for that diagnostic. The FVM force
  coefficients remain normalized with cylinder diameter times resolved span.

## Configuration and restrictions

Use Gaussian particles, laminar/DNS or no-SGS flow, `ViscousConfig.gbd(...)`,
`core_radius_ratio=1.0`, and `gbd_remeshing_kernel="M4_PRIME"`. The absolute
GBD threshold must be `omega_floor*h**2*span`, not the three-dimensional
`omega_floor*h**3`. Standalone inviscid operation also supports `scheme="NONE"`.
The coupled transfer method must be `buffered_m4_renewal`.

Set `panel_solver=None` and `bodies=()`. Use `vorticity_mixed` coupling with
FVM-owned pressure. **VPM pressure reconstruction and pressure-gradient
coupling are explicitly unsupported**, because their separate kernels still
implement three-dimensional induction. LineSampler and SurfaceSampler use
velocity/curl/derivatives and do not request that pressure path.

Three-dimensional regularization, divergence relaxation, axisymmetric
projection, LES, VLM, other diffusion/remeshing kernels, spanwise freestream,
and off-plane/transverse-strength sources are rejected. This operator does
not model finite-cylinder tip flow or three-dimensional wake instabilities.
Default bounds may retain the actual `[-0.5,0.5]` span; only `z=0` is populated.

The experiment launcher is `studies/panel_removal/run_cylinder.py`. Actual
cylinder runs remain conditional on the independent cube panel-removal gate.

## Continuous planar renewal

The interior transfer cutoff and GBD cutoff are distinct vorticity floors.
Each becomes a particle-strength threshold through `omega_floor*h**2*span`.
Renewal blends the interior threshold toward the GBD threshold as FVM
authority decreases. It uses continuous non-negative-garrote shrinkage,
`s = Gamma * max(1 - (threshold/abs(Gamma))**2, 0)`.

Planar renewal then recovers circulation and the two first moments using
only z-strength unknowns. With constraint rows `A=(1,x,y)` and desired
moments `b`, the correction is `W A.T (A W A.T)^-1 (b-A s)`, where the
diagonal metric is proportional to particle volume times `abs(s)`.
Normalizing the magnitude weights does not change this solution. Physical
particle volumes remain `h**2*span`; only the correction metric changes.
The implementation centres and scales the coordinates for conditioning,
rejects insufficient support, and preserves exactly zero transverse strength.

Binary neighbour redistribution is omitted in the planar branch. Otherwise
a newly retained node can receive a finite neighbour contribution and an
equal-volume share of the global correction even as its own shrunk strength
approaches zero. That discontinuity can prevent interface fixed-point
convergence. Magnitude-weighted recovery makes the newborn contribution
vanish continuously when the remaining support spans all three constraints.
The existing conservation and correction-fraction acceptance checks remain
active. The three-dimensional cube renewal path is unchanged.

Threshold-crossing regression tests exercise this complete renewal path at
`plane_z=0` and `0.137`, with amplitudes from `1e-8` to `1e8`. They require
proportionally shrinking output perturbations, exact circulation/first-moment
recovery, unchanged particle volumes, and rejection of rank-deficient support.
The historical map's finite jump is demonstrated independently; attributing
the original cylinder's 1.48 s plateau specifically to that jump still
requires the comparison with the corrected run on the identical native mesh.

## Verification

`tests/vpm/test_planar_induction.py` checks Lamb–Oseen velocity/curl, analytic
Jacobians by independent centred differences, source-centre limits, z
invariance, zero stretching, circulation and heat-kernel variance growth,
pruned six-moment recovery, renewal volume and normalization, rejection of
incompatible physics, real RK/GBD evolution, Gaussian dipole energy, and the
real LineSampler derivative/curl path.

`tests/coupler/test_planar_coupling.py` runs two native FVM/VPM coupling steps
on a small box with a persistent exterior dipole. It exercises mixed boundary
traces, renewal, diffusion, finite-state checks, spanwise consistency, and
the `h**2*span` storage contract. Both this integration and the real VPM test
have passed on CPU and native Metal. Metal qualification is explicitly
enabled with `OPENONDA_TEST_METAL=1` because sandboxed Metal access may fail
before the solver kernel is compiled.

The same test module also includes a real two-rank integration through the
public factory and PETSc FVM. It has passed with a root-only VPM, a persistent
exterior dipole, collective field gathering and clean shutdown on both ranks.
It uses span=2 rather than 1: h=.125 gives particle volume=.03125=h²L, checking
the span normalization explicitly. The maximum measured spanwise inconsistency
over its two exchanges was 1.77e-7 U∞. This remains a small integration test,
not a cylinder validation.

These tests establish operator and integration correctness. They do not
establish cylinder drag, shedding frequency, grid independence, or agreement
with the fully meshed reference; those require the gated physical runs.
