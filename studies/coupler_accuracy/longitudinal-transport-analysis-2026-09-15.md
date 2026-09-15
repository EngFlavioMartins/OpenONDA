# Longitudinal transport: continuum identity, particle limits, and a cube test

The proposed continuum correction is valid. It is not, by itself, a valid
particle-strength correction. The smaller next intervention is a
velocity-matched transfer test: two descriptions of the same velocity should
not change that velocity merely because their longitudinal vorticity differs.
No production code or simulation was changed for this analysis.

This note checks the interpretation of
[the cube mechanism study](cube-wake-mechanism-2026-09-15.md), against development
commit `fda69c0f`. The implementation uses positive-transpose stretching in
[`stretching_rate`](../../source/solvers/vpm/physics/induction/stretching.py).
The current represented-state transfer is in
[`blend_represented_state`](../../source/coupler/stable_renewal.py).

## The continuum statement

Let `w = omega_G`, `q = curl(u)`, `ell = w-q = grad(psi)`, and
`J_ij = partial_j u_i`. Vorticity and `J` have units `1/s`; `psi` has units
`m/s`. Let `K` denote unregularized Biot–Savart, and `P` and `Q=I-P` the
transverse and longitudinal Helmholtz projectors. Then `u=K w` and `q=P w`.
These global identities assume free space with suitable decay, or a periodic
domain with its zero/harmonic modes handled explicitly. A body potential can
be included locally in `u` and `J`, but its boundary problem cannot be ignored
when claiming a global velocity identity.

For incompressible inviscid flow, the proposed extension is

```text
D_t w = J.T w - 2 J.T ell
      = J.T q - J.T ell
      = J q   - J.T ell.
```

The last equality uses `(J-J.T)v = q cross v`, hence `(J-J.T)q=0`.
For a gradient, the product rule gives

```text
-(u . grad) ell - J.T ell = -grad(u . ell).
```

Consequently the Eulerian equation is

```text
partial_t w   = curl(u cross q) - grad(u . ell),
partial_t q   = curl(u cross q),
partial_t ell = -grad(u . ell).
```

The physical vorticity obeys Euler; `D_t psi=0` up to a spatially constant
gauge transports its gradient as a covector. The longitudinal part remains
invisible to velocity. Applying negative-transpose stretching to the entire
vorticity would instead destroy physical vortex stretching.

Uniform molecular diffusion can be included consistently by adding
`nu*Laplacian(w)` and using `D_t psi=nu*Laplacian(psi)`. This does not justify
an arbitrary variable-viscosity or LES operator: those require their own
stress/curl consistency. This note establishes neither a turbulence closure
nor a boundary-vorticity model.

## Why substituting Gamma/V is insufficient

For spherical Gaussian particles with normalized density `zeta_p`,

```text
w(x) = sum_p Gamma_p zeta_p(x-X_p),
Xdot_p = U_p.

D_t w(x) = sum_p zeta_p Gamma_dot_p
         + sum_p Gamma_p [(u(x)-U_p) . grad(zeta_p)]
         + sum_p Gamma_p sigma_dot_p partial_sigma(zeta_p).
```

With the native strength rate `Gamma_dot_p=J_p.T Gamma_p`, this becomes

```text
D_t w = J(x).T w + R_core,

R_core = sum_p zeta_p [J_p.T-J(x).T] Gamma_p
       + sum_p Gamma_p [(u(x)-U_p) . grad(zeta_p)]
       + sum_p Gamma_p sigma_dot_p partial_sigma(zeta_p).
```

This is an exact differentiation of the particle representation, not a
postulated subgrid model. The first two terms describe variation across a
finite core and transport by each centre's velocity; the last covers core
growth. They must not be dropped merely because particle quadrature is dense.
Extra numerical rates, if enabled, enter separately.

`Gamma_p/V_p` is a coefficient density. It is not the reconstructed
`w(X_p)=sum_r Gamma_r zeta_r(X_p-X_r)`, nor `q(X_p)`. Depositing
`-2 V_p J_p.T ell(X_p)` therefore produces approximately the **Gaussian
convolution** of the desired resolved source, with additional quadrature
error. An exact lift of that source into particle strengths would still
leave `R_core` in the represented equation.

Other necessary distinctions are:

- With nonuniform or evolving cores there is generally no single convolution
  that commutes with a Helmholtz projection.
- Particle volumes are quadrature weights, not an inverse of the Gaussian
  reconstruction. A continuum density derivation also has to account for
  changing weights; incompressibility makes material volumes constant only
  under its corresponding transport assumptions.
- A gradient field has nonlocal Helmholtz tails. Cropping it, excluding the
  solid, or multiplying it by a selection window creates curl unless the
  modification is made to its potential and differentiated consistently.
- Matching velocities at a few probes does not establish that a fitted
  particle component is globally longitudinal. Derivatives and independent
  off-grid points also need checking.
- A finite set of isotropic vector blobs need not represent the exact
  Helmholtz components on that same set of centres. A stable resolved-band
  approximation must be qualified rather than assuming exact deconvolution.

For a common fixed radius, the distinction is especially clear. Write
`w=G_sigma a`, where `a` is the coefficient-density continuum limit and
`G_sigma=exp((sigma^2/4)*Laplacian)` for this Gaussian convention. Then

```text
L a = -(u . grad) a + J.T a,
partial_t w = G_sigma L a,
R_core = (G_sigma L - L G_sigma) a.
```

For smooth fields this commutator is `O(sigma^2)` as `sigma` decreases.
Refining quadrature at fixed `sigma` does not remove it. Finite filtering is
also central to filtered VPM formulations; see the explicitly filtered
governing-equation approach of
[Alvarez and Ning (2023)](https://scholarsarchive.byu.edu/facpub/7123/).
The identities in this note are derived above, not attributed to that paper.

One minimal extension of the **coefficient-density** equation is

```text
a_T = P a,     a_L = Q a,
D_t a = J.T a - 2 J.T a_L.
```

At fixed common core this removes the velocity-rate dependence on `a_L`:
its contribution is `-grad(u . a_L)`, and convolution preserves a gradient.
But `a_L=G_sigma^(-1) ell`, not `ell`. Nor does the remaining transverse
dynamics become exact Euler at finite core: generally
`J.T a_T != J a_T`, because `curl(u)=G_sigma a_T`, not `a_T`.
Thus this extension addresses one defect of representation dependence while
retaining the native finite-core dynamics of the transverse density. It is
a candidate mathematical extension, not a qualified discrete implementation.

## What has actually been established

The existing injected-gradient experiment establishes that positive-transpose
transport can make an initially velocity-invisible component active. Its
affine-strain control removes body, recirculation and time-integration
explanations; its quadrature refinement removes ordinary quadrature error as
the explanation for that particular nonzero limiting rate.

It does not prove inconsistency of classical VPM for a physically solenoidal
initial condition in a joint particle/core convergence limit. Euler only
prescribes evolution on the solenoidal constraint; an extension away from
that constraint is an additional choice. Nor does the injected mode establish
what fraction of the saved cube's velocity error comes from its actual
longitudinal component. The safer conclusion is **demonstrated sensitivity to
non-solenoidal representation, with case-specific causation still requiring
an actual-state budget**.

The new actual-state nodal-source measurements supplied by the Coupler task
are stronger evidence of relevance: they alter the signed asymmetry-growth
rate. They remain approximate interventions of the extra-filtered kind
described above. Their 48%/29% reduction near the seam cannot be presented as
an exact percentage of longitudinal-error causation; their much smaller
effect elsewhere further argues against a universal one-term explanation.

## Independent bounded check

[`longitudinal_transport_check.py`](longitudinal_transport_check.py) evaluates
finite, fully resolved Fourier series in a periodic three-dimensional box.
The velocity is nonuniform, incompressible and self-consistent with `q`.
Both input and product modes are resolved on 24-cubed and 32-cubed grids;
there is no particle quadrature, time integration or body model. The
[JSON evidence](results/cube-wake-cause-2026-09-15/longitudinal-transport-identities.json)
records all six radius/grid combinations.

At `sigma=0.2 m`, the 32-cubed result is:

| Measurement | RMS |
| --- | ---: |
| Resolved continuum corrected vorticity-rate identity error, 1/s² | 8.45e-13 |
| Native longitudinal velocity rate, m/s² | 0.670800 |
| Coefficient-gradient correction residual, m/s² | 3.14e-16 |
| Substituting resolved `ell` into coefficient correction, m/s² | 0.034408 |
| Transverse-only finite-core velocity-rate error against Euler, m/s² | 0.256057 |

The last error is `0.900354`, `0.256057`, `0.066175` for radii `0.4`, `0.2`,
`0.1 m`: consistent with approaching second-order core error. The two grids
agree to roundoff in these physical rates. Removing longitudinal leakage
therefore does not remove finite-core error, even in this deliberately simple
smooth problem. This is a manufactured mathematical check, not cube evidence.

## A conclusive actual-state evolution budget

At one saved cube time, freeze transfer, diffusion, pruning, and body geometry.
Use the same physical velocity and derivative convention throughout. Evaluate
the full particle representation and its exact instantaneous derivative,
including centre motion, with the original strengths and cores. The identity
above gives

```text
u_dot_native = K curl(u cross q)
             + 2 K(J.T ell)
             + K R_core.
```

This separates physical Euler evolution, resolved longitudinal leakage and
finite-core transport error without substituting `Gamma/V` or deconvolving
the saved field. A changing boundary potential or additional native operators
must be retained as their own contributions; the displayed equation is the
particle-induced velocity-rate budget for the specified inviscid stage.

An isolated implementation should:

1. Read the actual complete `t=6` or `t=8` state. Sample `w`, its derivatives,
   `u`, `J`, and the derivative of the actual moving Gaussian sum on a padded
   three-dimensional grid; validate selected values by independent direct
   sums. Do not substitute a newly invented gradient perturbation.
2. Use one compatible Helmholtz/Biot–Savart operator to form the three
   contributions. Verify padding and spatial refinement independently, and
   specify the treatment of body and harmonic fields. Merely zeroing values
   inside the solid or truncating `ell` at the seam invalidates the identity.
3. Close the sum against the independent native direct velocity-rate kernel
   at the existing 392 probes and additional off-grid points. Report signed
   contributions to asymmetry growth and to the reference-error derivative,
   not only each contribution's RMS magnitude.
4. If the actual longitudinal term is resolved above this numerical uncertainty,
   test the corresponding correction in a separate representation on a single
   inviscid step and two half steps, recalculating its split and `J` at each
   temporary stage. Check that reconstructing the unchanged full state first
   reproduces its velocity, gradient and native rate. Failure of this control
   prevents a claim about the correction.

The gradient mechanism is not an adequate dominant-error explanation if its
actual signed contribution is small, has the wrong direction, or is below
the reconstruction/padding uncertainty. A large `K R_core` identifies a
different obstacle; reducing only the nodal longitudinal source would not
address it. One frozen budget and a short step check establish an instantaneous
mechanism, not long-time accuracy or nonlinear stability.

## The simpler transfer correction and its falsification test

The continuum velocity-consistent incremental transfer is

```text
delta_w = curl[eta (u_F-u_V)]
        = eta (q_F-q_V) + grad(eta) cross (u_F-u_V),
w_new   = w_old + delta_w.
```

It preserves `div(w)` exactly at continuum level. In the same all-space
velocity convention it induces
`u_new=u_V+P[eta*(u_F-u_V)]`: the incompressible projection of the intended
velocity blend. It preserves existing longitudinal contamination rather than
removing it, and a body boundary response remains a separate requirement.

If `u_F=u_V`, this update is identically zero. In contrast, even with a
perfect donor `q_F=q_V`, the current ideal vorticity blend changes the field
by `-eta*ell`. Its curl is `-grad(eta) cross ell`, generally nonzero. It can
therefore create a velocity change from agreement between the two physical
velocities. This is a more direct transfer consistency failure than showing
that a generic divergence metric increases.

The bounded test is the parent's proposed gradient/null-velocity case through
the actual production blend, followed by an **actual-cube velocity-matched
donor control**. Use constant-authority controls and quadrature refinement.
For the cube, first supply the actual analytic VPM velocity and its same-kernel
curl as an oracle donor, including the contribution of preserved outer
particles. Then test the FVM sampling/reconstruction path separately. This
distinguishes transfer's target equation from errors in donor gradients and
reconstruction. Measure the native before/after velocity and rate, with the
moment correction, masking and pruning stages separately visible.

For a discrete correction, form `curl_h` with a matching `div_h` so that
`div_h curl_h=0`, before addressing its Gaussian reconstruction. On a complete
uniform lattice with a common scalar convolution and compatible boundary
handling, convolution commutes with these difference operators; finite
truncation and masking can break that property. Depositing `V*curl_h[...]`
also filters the intended velocity increment. A later mask or componentwise
pruning can destroy its divergence guarantee. Those effects must be checked
through the final native particle field, not just the pre-deposition lattice.

The velocity-matched transfer test should precede development of a new stage
law. It has an exact physical fixed point, uses the current transfer owner,
and can expose an avoidable source of longitudinal pollution without conflating
it with finite-core evolution. Neither passing that test nor the continuum
proof is sufficient to qualify the developed coupled cube wake.

## Implemented study prototype and its qualification

[`cube_curl_residual_transfer.py`](cube_curl_residual_transfer.py) now provides
the pure `curl_residual_correction` function. It returns increments on a
complete regular lattice and changes no solver. Its callbacks receive the six
arrays `X +/- h e_j`, with velocity in m/s and scalar weights in `[0,1]`.
The existing velocity-trace integral is evaluated with spacing `2h` and divided
by eight, giving `h^3 curl_h` with the centered nodal derivative. This is
deliberate: the original face-centre `h/2` stencil has a different derivative
symbol from the nodal centered divergence and would not cancel it exactly.

The insertion point for the bounded cube experiment is:

1. Form a coefficient-preserving baseline, complete its ordinary remapping,
   masking, pruning and moment repair, and retain the outer particles.
2. Evaluate the **post-remap** VPM velocity, including its body/background
   contribution, at the same points as the FVM trace.
3. Form the masked velocity residual inside the curl and add the returned
   correction on its complete support. Do not apply the old vorticity blend
   first. Do not mask, prune or repair the correction afterwards. Coincident
   coefficients can be combined only when they use the same kernel and core.

In particular, do not restore the old impulse after adding the physical
increment. For compact support, its continuum linear impulse divided by
density is `delta(I/rho)=integral eta*m*(u_F-u_V) dV`, in m^4/s, although its
integrated vector strength is zero. Subsequent native GBD remains a separate
operator with its own measurements.

The default does no inverse filtering. The optional scalar step `(2I-G_h)d`
acts on the increment alone and reduces Gaussian reconstruction error while
preserving the matching discrete divergence on complete support. It requires
an additional six-core halo. It also spreads support: the manufactured test
shows nonzero strengths appearing at forbidden solid centres at coarser
resolutions, even though the raw curl's guarded support excludes them. This
is why the first cube experiment should use `deconvolution_steps=0`. A body
mask belongs inside the curl, and its zero-residual guard must cover the
stencil of every excluded centre; post-curl clipping is not a remedy.

[`qualify_cube_curl_residual_transfer.py`](qualify_cube_curl_residual_transfer.py)
uses a compact three-dimensional authority, a smooth spherical fluid mask,
and an analytical incompressible velocity/curl pair. The
[five-level evidence](results/cube-wake-cause-2026-09-15/curl-residual-convergence-qualification.json)
passes all ten gates. Matched velocities give exact zero increments; the
largest matching-divergence residual is `3.97e-13` without inverse filtering
and `7.52e-13` with it. On the final refinement, observed orders are `1.79`
for the analytical curl, `1.93` for reconstructed Gaussian vorticity, and
`1.79` for its continuous divergence. These are bounded manufactured results,
not a cube run. Earlier coarse-grid gate failures remain in the evidence
directory; neither the reference nor the thresholds was weakened.

The fixed `sigma/h=1.1` series is **not** an asymptotic strong-divergence
qualification. Gaussian lattice aliases scale approximately as
`exp(-pi^2*(sigma/h)^2)` in amplitude, while their spatial derivatives bring
a factor `1/h`; indefinitely refining at a fixed ratio need not suppress
that error. The separate
[fixed-core quadrature check](results/cube-wake-cause-2026-09-15/curl-residual-fixed-core-qualification.json)
holds `sigma=0.11 m` and uses `h=0.1, 0.05, 0.025 m`. Continuous Gaussian
divergence RMS decreases from `0.0927261` to `0.0255795` to `0.00656149 1/(m s)`,
with observed orders `1.858` and `1.963`. This is the appropriate separation
of quadrature/derivative consistency from a fixed physical filter width.

The [API checks](results/cube-wake-cause-2026-09-15/curl-residual-api-checks.json)
cover hover examples, the six-query order, read-only inputs, invalid weights,
missing inverse radius, incomplete support rejection, and reproduction of
the qualified `h=0.1` case. Ruff and targeted Pyrefly checks also cover the
two new modules. Importing the existing coupler produced an MPI socket-bind
warning in this sandbox; the serial numerical checks completed successfully.

The unchanged-ratio short cube replay remains scientifically useful as a
controlled intervention. Exact velocity agreement preservation does not
depend on the core ratio. Keeping the ratio, GBD and health gates fixed
therefore tests the transfer change under the actual case conditions, while
continuous Gaussian divergence, longitudinal content and reference error
remain measured limitations. It must not be described as a converged
divergence-free particle method or a qualified long-time coupled solution.

## Actual t=6 evolution budget

[`cube_represented_evolution_budget.py`](cube_represented_evolution_budget.py)
now evaluates the actual 43,479-particle saved state. It advances no simulation.
The [four-case budget](results/cube-wake-cause-2026-09-15/represented-evolution-budget/budget.json)
uses quadrature spacings `0.03` and `0.02 m`, and periodic boxes with lengths
`(7.2,3.6,3.6)`, `(9.6,4.8,4.8)`, and `(10.8,5.4,5.4) m`. Saved coordinates
are within `2.67e-7 m` of their GBD lattice; the common radius is `0.066 m`.
Native stage values are the saved float32 rates, accumulated here in float64.

The three resolved terms use this actual cloud's self-induced velocity plus
the unit freestream. Native-minus-self motion/stretching is a separate fourth
term. It includes the body, native induction discrepancies, and the periodic
self-field approximation. This avoids silently continuing a panel potential
through the solid, where the simple global divergence-free identity would
require additional boundary/source terms.

On the largest box, the instantaneous derivative of reflection-odd velocity
energy, in m²/s³, is:

| Contribution | Renewal seam | Outer wake |
| --- | ---: | ---: |
| Native free-space direct rate | 0.00986705 | 0.00122671 |
| Self-flow Euler term | 0.00416202 | 0.00120850 |
| Resolved longitudinal term | 0.00627055 | 0.00053208 |
| Finite-core/transport term | -0.00062329 | -0.00048809 |
| Native-minus-self term | 0.00005743 | -0.00002561 |

The longitudinal term accounts for **63.55% of the instantaneous seam growth**;
the finite-core term partly opposes it. This is not a percentage of accumulated
velocity error. In the outer wake, longitudinal and finite-core contributions
nearly cancel. Also, `R_core` depends on the full represented field: it must
not be interpreted as an independent error of the transverse component alone.

The final domain enlargement changes the seam longitudinal and core terms by
only `0.093%` and `0.12%`. Refining quadrature from `0.03` to `0.02 m` changes
the seam longitudinal contribution by `3.2e-8 m²/s³`; the independently formed
product-rule closure improves to `6.7e-9 m/s²` RMS. The largest-domain periodic
native rate differs from the free-space direct rate by `0.000613 m/s²` RMS,
and its seam energy derivative differs by `3.1e-7 m²/s³`.

The absolute periodic self velocity still differs from free space by
`0.0202 m/s` RMS on that largest box. Thus the robust longitudinal/core signed
conclusion does **not** certify a converged free-space Euler/body partition.
All source hashes remain unchanged. A 27.2-MB scalar Fourier potential and its
mean coefficient density are retained in the ignored result directory; their
path, hash and reconstruction convention are in the report. They specify a
periodic longitudinal/transverse coefficient split. Cropping it into a
free-space cloud or masking the solid is not a velocity-preserving projection.

The smallest next real-cloud intervention is the already measured nodal
covector source, recomputed at every actual RK temporary state, for a short
`t=6` to `6.1` replay with the same particle population and original physical
operators. Its initial velocity is unchanged and its frozen-stage effect is
known; its extra filtering and incomplete global support remain explicit
approximations. A global projection is not presently a cheaper, qualified
replacement: its noncompact tails and body treatment must first pass an
independent free-space velocity/gradient check. The short replay must establish
useful accuracy at comparable cost before either candidate becomes production
code. The rejected curl-transfer trajectory is separate evidence and is not
relabelled as a successful fix.
