# Acceptance review of the nodal covector control

The completed 100-step comparison improves measured velocity error by 11.4%
near the interface and 15.7% in the outer wake, at essentially unchanged
particle count. It does not justify production use: a frozen self-flow audit
demonstrates a material occupied-support defect in the added source.

The [completed report](cube-transfer-cause-2026-09-15.md) gives seam velocity
error `0.114856 U_inf`, versus `0.129684 U_inf` for the native control, and
outer error `0.0865063` versus `0.102587 U_inf`, at `t*=7`. Downstream reflection
asymmetry falls from the native `0.0728906` to `0.0499573 U_inf`. The final
populations are 51,300 and 51,313, respectively, and the original health gates
pass. This endpoint uses an actual saved reference state. The experiment
prescribes reference donors from `t*=6` to `7`; it is not a new coupled
trajectory or evidence of force accuracy or long-time agreement.

## The net-strength source is not negligible by default

The [earlier ten-step report](results/cube-wake-cause-2026-09-15/covector-control-t6/replay.json)
has the following last instantaneous correction:

```text
sum C_p = (0.0105888, -0.00382155, -0.00756738) m^3/s^2,
norm(sum C_p) = 0.0135644 m^3/s^2.
```

This is only `0.618%` of the sum of individual correction-rate magnitudes,
but `69.8%` of the reported native net-strength-rate norm. Cancellation-based
normalization alone is insufficient to establish that the residual is benign.
That report's last measurement comes from an unused gradient-refresh call;
it is not the correction integrated by the RK step.

For `ell=grad(psi)`, `J_ij=partial_j u_i`, and incompressible flow, the intended
resolved source is `C=-2 J.T ell`. Integration by parts gives

```text
integral_Omega C dV = -2 integral_boundary psi (J.T n) dS,

(1/2) integral_Omega x cross C dV
    = - integral_Omega psi curl(u) dV
      - integral_boundary psi x cross (J.T n) dS.
```

In suitably decaying free space the net-strength source is zero, while the
impulse source is generally nonzero. The latter can remove an unphysical
force contribution from the original off-constraint evolution. Do not force
it to zero. In a body-bounded fluid domain, the displayed surface terms must
be considered before asserting zero net source.

A normalized, centered Gaussian preserves zero and first moments. Extra
Gaussian filtering therefore does not itself explain a nonzero net-strength
source. Occupied-support truncation, quadrature, body treatment and derivative
errors need to explain the difference from the applicable moment target.

## The accepted-stage ledger closes, but does not validate the source

The [100-step ledger](results/cube-wake-cause-2026-09-15/covector-control-ledger/replay.json)
records 200 actual Heun stage evaluations, starting with distinct stages zero
and one, and exact particle invariants around RK, GBD and transfer. Signed
correction rates use the actual one-half weight of each Heun stage. The maximum
RK net-strength closure norm is `1.18e-8 m^3/s`. The integrated added source is
`(0.00220819, 0.00577868, 0.00127855) m^3/s`; its integrated impulse source is
`(0.00223835, -0.0108015, 0.00611430) m^4/s`. This establishes bookkeeping
consistency, not physical consistency of those source terms.

The RK-weighted impulse-source integral and the exact accepted impulse change
are distinct: impulse also involves particle motion and is bilinear in
position and strength. The net-strength contribution is a linear RK budget.

The [actual ten-step particle invariants](results/cube-wake-cause-2026-09-15/ten-step-particle-invariants.json)
show corrected-minus-native impulse differences of approximately
`(-0.000370,-0.000681,-0.000426) m^4/s`, and net-strength differences of
`(0.006563,-0.001066,0.0000362) m^3/s`. Neither is automatically a violation in
this transferred finite cloud, but their smallness relative to the large
streamwise impulse does not establish small transverse-force errors. Compare
the appropriate components and physical reference loads. Reducing the
magnitude of one pre-existing net-strength component is not a conservation
proof.

Host downloads return independent arrays, so uploading the corrected rate
does not overwrite the recorded native rate.

## Production acceptance remains conditional

A successful 100-step comparison is a screening result. Promotion requires
sustained error improvement at comparable cost, a closed accepted-stage and
exchange budget, and source-moment consistency under appropriate support and
quadrature refinement. The previously established fixed-core refinement is
relevant; indefinitely refining at fixed `sigma/h` is not a general strong-
divergence convergence argument.

An unexplained secular net-strength change, compensation by repeated renewal,
or worsened force prediction would block promotion even if the wake image is
cleaner. Keep the existing health limits and reference conditions. A global
moment repair must not be added merely to make this audit pass. A fresh coupled
trajectory and appropriate numerical/backend qualification remain necessary
before claiming a general solver fix. No new experiment was run for this review.

## A compact potential control is defensible, but remains unqualified

The [support budget](results/cube-wake-cause-2026-09-15/covector-support-budget/support.json)
separates the periodic self-flow source integral, the cube boundary contribution,
and existing-particle quadrature. On the `9.6 x 4.8 x 4.8 m` box the whole-periodic
integral is at roundoff. Excluding the cube gives approximately
`(-0.001962, 0.00003949, -0.00005007) m^3/s^2`, whereas occupied-particle quadrature
gives `(0.021190, -0.010782, -0.004401) m^3/s^2`. Refining the quadrature from
`0.03` to `0.02 m` on the smaller box leaves a comparable discrepancy. This
control establishes an occupied-support defect; it does not supply the complete
native body-flow budget.

One bounded candidate is to localize the longitudinal **potential** and integrate
its source through shared cell faces. For a dimensionless smooth cutoff `chi`,
let `phi=chi*psi`, where `ell=grad(psi)`. In incompressible fluid,

```text
C_chi = -2 div(phi J.T)
      = -2 J.T grad(phi)
      = chi*C - 2*psi*J.T grad(chi).
```

Here divergence contracts the second tensor index: component `i` is
`partial_j(phi*partial_i u_j)`. The compact gradient `grad(phi)` remains
velocity-invisible under the free-space Biot--Savart operator when `phi`
extends smoothly by zero. At the resolved continuum level, the source removes
the native amplification associated with this selected gradient component.
It leaves `grad[(1-chi)*psi]` uncorrected. The cutoff-gradient term is an
essential part of this different source, not a repair of the original one.

For a cell represented by particle strength `Gamma_i`, use the integral rate

```text
Gamma_dot_i^correction = -2 sum_faces A_f*phi_f*(J_f.T n_if).
```

Face area is in `m^2`, `phi` in `m/s`, `J` in `1/s`, and the strength rate
in `m^3/s^2`. Compute each internal face flux once and apply opposite signs
to its two cells. With zero external flux the net rate then vanishes up to
applied-arithmetic roundoff, independently of discrete derivative accuracy.
No additional cell-volume factor belongs after the face integral. For compact
smooth support the continuum impulse-source target is
`-integral phi*curl(u) dV`, in `m^4/s^2`; it is generally nonzero.

The following limits are part of the candidate's definition:

- **The cutoff needs physical support.** Resolve its transition over a fixed
  physical width inside the occupied fluid region. It must vanish before the
  outer boundary, internal holes, and the body. Setting only the outer face
  flux to zero while retaining `chi=1` up to that face instead creates a sharp
  surface source, with cell source density scaling as `1/h`. A fragmented
  cloud may not have room for a useful smooth cutoff. A cutoff reaching the
  body requires its boundary term; zero flux must not become an invented wall
  condition. Body strain may enter `J` where the actual velocity is
  incompressible and the potential decomposition holds.
- **Localization fixes a gauge.** Replacing `psi` by `psi+c` changes this source.
  For the free-space Gaussian particle field, choose `psi -> 0` at infinity.
  With `r_p=x-X_p`, the existing radial induction factor gives

  ```text
  f_sigma(r) = [erf(r/sigma) - 2*(r/sigma)*exp(-(r/sigma)^2)/sqrt(pi)]
               / (4*pi*r^3),
  psi(x) = sum_p f_sigma_p(|r_p|)*(Gamma_p dot r_p).
  ```

  This follows from the same Gaussian vector potential as the velocity
  `sum_p f_sigma_p*(Gamma_p cross r_p)`, so the scalar sum can share induction
  interactions in principle. It is the potential of the represented Gaussian
  field. If nonparticle velocity is present, identifying its gradient with
  `omega_G-curl(u)` also requires that contribution to be curl-free in the
  fluid region. A periodic coefficient potential is not interchangeable with
  this free-space resolved potential.
- **Conservation does not establish consistency.** In a compressible velocity
  field the flux form contains the additional term
  `-2*phi*grad(div(u))`. Discrete face sampling and product approximation also
  affect its relation to `-2 J.T grad(phi)`. Measure `D_h(J.T)` using those
  same faces; a small velocity divergence alone does not bound its gradient.
  Compatible derivatives help, and fixed-core spatial refinement must assess
  the remaining error.
- **The cells must belong to the evaluated stage.** The current RK source uses
  advected temporary particle positions. Attaching a frozen Cartesian face
  stencil to displaced particles preserves an algebraic sum but does not
  automatically define a consistent physical finite-volume divergence. Until
  stage geometry is specified, this is a frozen-lattice control rather than
  an established RK source.
- **Gaussian representation still filters the rate.** At common fixed core,
  depositing the cell-integral rates into particles has the refined-quadrature
  target `G_sigma*C_chi`, not `C_chi`. The normalized convolution preserves
  the continuum net strength and first moment but not exact resolved
  covector cancellation. This approximation must remain explicit.

The smallest useful qualification consists of one manufactured flux case and
one frozen-state gate, without advancing a solver:

1. On three small Cartesian grids, use the analytic incompressible velocity
   `u=(a*x-Omega*y, Omega*x-a*y, 0)` and compact potential
   `phi=phi_0*b(x/L)*b(y/L)*b(z/L)`, where `b(s)=(1-s^2)^4` for `abs(s)<1`
   and zero otherwise. The constant `J` makes `D_h(J.T)=0` exactly away from
   floating-point roundoff, isolating face conservation and source accuracy.
   Compare the integral rates with `C=-2 J.T grad(phi)`, check zero net
   strength, and verify the nonzero impulse target
   `(0, 0, -2*Omega*phi_0*L^3*(256/315)^3)`. Require spatial convergence
   without shrinking the physical support or any Gaussian core used for an
   induced-velocity check. This tests the flux operator, not a complete
   Gaussian particle transport law.
2. At the saved `t*=6` lattice, first check whether existing occupied cells
   admit a resolved smooth taper around the region of interest, wholly in
   fluid. If so, evaluate the free-space resolved potential and source once.
   Compare actual induced velocity-rate and impulse against the localized,
   Gaussian-filtered target, retaining the cutoff-gradient contribution
   separately. Measure the applied net-strength rate and `D_h(J.T)` on the
   same faces. Finer integration of this same frozen target may assess
   quadrature without adding evolving particles. This cannot by itself
   establish convergence of the unchanged coarse occupied cloud.

If the compensation removes the benefit, or the occupied cells cannot
accommodate the taper, reject the candidate. Neither suitable support nor a
consistent moving-stage cell geometry has been established here, so an
inexpensive RK implementation cannot be promised. A compact curl source is
not an exact general substitute: preserving a source's induced velocity
requires its Helmholtz projection, which is generally noncompact. There is
no validated small full-body correction at present. This assessment adds no
implementation or simulation.
