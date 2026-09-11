# Stability of the coupled Gaussian vortex-particle update

Date: 2026-09-10  
Scope: fixed-`N`, fixed-positive-core, direct-induction inviscid VPM ODE  
Status: accepted by Boss as a bounded accuracy/conditional-perturbation study; no production changes

## Result

A useful but deliberately limited result is available.

1. **The mathematical regularized finite-particle ODE is locally well posed.**
   For a fixed number of particles and pair cores bounded below by
   `sigma_min>0`, the exact Gaussian Biot--Savart field and all of its spatial
   derivatives have finite analytic limits at particle coincidence.  Its
   coupled position/strength vector field is therefore smooth and locally
   Lipschitz.  This removes a collision singularity, but it does not prove
   global-in-time bounded vortex strength: the stretching equation is
   quadratic in the strengths.  The literal production evaluator adds a
   small unmatched branch splice at `rho=0.2`, so this theorem applies to it
   only away from that switching hypersurface.
2. **The spatial and temporal questions must not be collapsed into one
   “stability” claim.** Core overlap, fill distance, separation, mesh ratio,
   quadrature moments, and particle disorder govern spatial consistency and
   the constants in the particle ODE.  They do not by themselves establish
   contractivity of a time integrator.  Conversely, a third-order time update
   does not repair a poorly sampled vortex field.
3. **Coupled SSPRK3 has a rigorous finite-step perturbation upper bound and
   third-order accuracy, but no VPM-specific SSP theorem was found.** The SSP
   label only transfers a forward-Euler monotonicity property after that
   property has been proved in a specified functional.  No such
   nonexpansive functional is known here for general three-dimensional vortex
   stretching.
4. **The historical advection-then-stretching path has a first-order split
   defect.** Its leading local defect is a position/strength commutator that
   becomes increasingly sensitive to small cores and clustered strengths.
   A position-strength-position composition removes that leading defect and
   is second order; the unsplit coupled SSPRK3 update is third order.
5. **The actual-kernel probes support the accuracy result, not a universal
   stability threshold.** Across 12 `N=27` combinations of regular, two seeded
   jittered, and clustered/voided clouds with `sigma/ell` in `{0.6,1.0,1.5}`, the
   observed finest-pair orders were 2.974–3.000 for coupled SSPRK3,
   0.99993–1.00010 for the historical split, and 1.99995–2.00007 for the
   symmetric split.  Centered-difference tangent estimates showed
   physical/nonnormal growth but no material extra amplification from coupled
   SSPRK3 at the tested step.

The evidence therefore supports keeping the common-stage coupled update on
accuracy and conservative perturbation-bound grounds.  It does **not** prove
global nonlinear stability, a universal `sigma/ell` threshold, spatial
convergence of the complete production method, or stability of a driven VLM
wake with diffusion, LES, core evolution, particle lifecycle changes, and an
approximate induction backend.

## 1. Model actually studied

For particle positions `X_i`, vector circulations `Gamma_i=omega_i V_i`, and
fixed core radii `sigma_i>0`, define

```text
r_ij       = X_i - X_j,
sigma_ij   = (sigma_i + sigma_j)/2,
rho_ij     = |r_ij|/sigma_ij,
zeta(rho)  = pi^(-3/2) exp(-rho^2),
q(rho)     = [erf(rho) - (2/sqrt(pi)) rho exp(-rho^2)]/(4 pi),
K_sigma(r) a = q(|r|/sigma) (a x r)/|r|^3,
U_i        = sum_j K_sigma_ij(r_ij) Gamma_j,
J_i        = grad U(X_i),
S_i        = J_i^T Gamma_i,
Y'         = F_sigma(Y) = (U(X,Gamma), S(X,Gamma)).
```

The centre value is `K_sigma(0)=0`; because the exact kernel has `q(rho)=rho^3/(3
pi^(3/2))+O(rho^5)`, the centre velocity gradient is finite.  The probe uses
the production `rho<0.2` series and Abramowitz--Stegun `erf` approximation,
the production pair core, and the historical/default `TRANSPOSED` strength
law.  At `rho=0.2`, the device formula evaluated with the written constants in
64-bit arithmetic has a branch jump `1.214e-8` (`2.597e-5` relative to the
exact Gaussian value); direct 32-bit emulation gives `5.210e-9`
(`1.114e-5` relative).  The host exact-`erf` registry still splices the same
truncated series and has a smaller `1.689e-9` jump (`3.612e-6` relative).
Thus the literal evaluators are piecewise smooth rather than globally smooth,
and the device gradient formula, which uses the exact Gaussian
`q`--`zeta` identity, is not the exact derivative of its approximate `q`
branch.  A centered-difference check had maximum relative operator mismatch
`3.15e-5` near that crossover and `2.40e-12` at the origin.
See [`kernel_crossover.csv`](vpm-stability-theory/kernel_crossover.csv) and
[`derivative_checks.csv`](vpm-stability-theory/derivative_checks.csv).

### 1.1 Kernel-splice source defect candidate

This is a precise source defect candidate, not an attributed rotor failure.

- The host registry `source/solvers/vpm/kernels/base.py:387–399` uses exact
  `math.erf` above the switch but the truncated series below it; its left and
  right limits therefore differ at the switch.
- The shared device factory `source/solvers/vpm/kernels/gaussian.py:15–86`
  uses the Abramowitz--Stegun approximation and is instantiated for direct
  induction in both `f32` and `f64`; the fixed approximation coefficients mean
  selecting `f64` does not remove the formula-level splice.  The production
  FMM consumes this same device `q` but currently permits only `f32`.
- The LBVH treecode carries a separate `f32` copy at
  `source/solvers/vpm/physics/induction/treecode/lbvh.py:1140–1171` using the
  same shared crossover/constants.  `config/constants.py:95–107` explicitly
  requires the direct and LBVH series copies to stay synchronized.

The values above are formula-level NumPy evaluations with explicit `float32`
rounding and with 64-bit arithmetic, not a device-by-device Taichi conformance
test.  A production fix should match at least value and first derivative
across all three paths and add `f32`/`f64` cross-backend tests.  No production
edit was authorized or made in this study.

This convention is mapped line by line in
[`equation-map.md`](vpm-stability-theory/equation-map.md).  In particular, the
current solver evaluates position and strength at common Runge--Kutta stages
while keeping cores fixed within that RK call, then applies its configured
diffusion/core operation outside the inviscid call.  The historical parent
`bb1718c9` defaulted to a full advection SSPRK3 step followed by a full
stretching SSPRK3 step.

The analysis does not include freestream or VLM time dependence, viscosity,
core spreading, LES, relaxation, remeshing, insertion/deletion, tree/FMM
error, or accepted-step rejection/health logic.  Those exclusions are
essential: they are not small notational variations of the autonomous
fixed-state-space ODE.

## 2. Geometry, overlap, and disorder

Several different lengths are needed; using “particle spacing” for all of
them hides failure modes.

- `ell_i=V_i^(1/3)` is the volume-equivalent spacing.  It is also used as the
  LES filter width in the current solver, but it is not the core radius.
- `d_i=min_{j!=i}|X_i-X_j|` is the nearest-neighbour distance.  The local
  overlap ratios are `eta_V,i=sigma_i/ell_i` and
  `eta_nn,i=sigma_i/d_i`.
- On a stated region `Omega`, the fill distance is
  `h_Omega=sup_{x in Omega} min_i |x-X_i|`; the separation radius is
  `q_X=(1/2)min_{i!=j}|X_i-X_j|`; `h_Omega/q_X` is the mesh ratio.  A bounded
  mesh ratio is a quasi-uniformity condition; a single median overlap ratio
  cannot reveal a void next to a cluster.
- For a diagnostic point `x`, define Gaussian quadrature weights
  `w_j(x)=V_j sigma_j^(-3) zeta(|x-X_j|/sigma_j)`.  The zeroth-moment defect is
  `m_0(x)=sum_j w_j(x)-1`, the scaled first moment is
  `m_1(x)=sum_j w_j(x)(X_j-x)/sigma_j`, and the eigenvalues of
  `sum_j w_j [(X_j-x)/sigma_j][(X_j-x)/sigma_j]^T` describe directional
  conditioning.  Coverage, moment defects, anisotropy, clustering, and voids
  are complementary—not interchangeable—diagnostics.

### 2.1 What the primary convergence theory actually says

Cottet’s original three-dimensional grid-free analysis defines the particle
ODE and regularized velocity in equations (5.7)–(5.9), explicitly notes that
the nonlinear finite strength system gives only local existence without
further estimates, and then proves a continuous-time spatial convergence
result under strong hypotheses.  Theorem 5.2 assumes a smooth normalized
regularizer, vanishing moments through the requested approximation order,
finite higher moments, and

```text
h <= C epsilon^(1+s),    s>0,
```

where `h` is particle discretization scale and `epsilon` is smoothing scale.
It obtains velocity error `O(epsilon^d)` on a smooth-Euler time interval.
This is an asymptotic *increasing-overlap* condition (`epsilon/h` grows as the
discretization is refined), not a statement that an arbitrary finite cloud is
safe whenever `sigma/h` exceeds one.  See pp. 269–271 of the
[primary paper](https://ems.press/content/serial-article-files/17460).

The production Gaussian is smooth, normalized, rapidly decaying, and has
zero first moments, but it has a nonzero second moment.  Thus it satisfies the
moment cancellation in that theorem only for the low-order `d=2`
specialization; the theorem cannot be quoted as arbitrary-order convergence
for this kernel.  It also does not analyze discrete time integration.

Winckelmans’ original dissertation independently records the regularized
particle equations, the classical/transposed/mixed strength laws, and the
fact that particle vorticity need not remain solenoidal even though the
induced velocity is divergence free (pp. 62–74 of the
[primary thesis](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf)).
That is another spatial/representation consistency issue; it is not cured by
changing RK stages.

### 2.2 Finite-cloud evidence

The probe used a `3x3x3` cloud (`N=27`) with unit volumes, two deterministic
18%-jitter realizations, and a deliberately compressed clustered/voided
cloud.  The exact fill distance is a continuum supremum; the probe only took
the maximum on a `9x9x9` grid in `[-1,1]^3`.  Its reported fill distance and
mesh ratio are therefore sampled lower estimates/proxies, not certified
continuum values.  Kernel moments were measured at all particle sites and
therefore include finite-cloud boundary truncation.

| Cloud | `sigma/ell` | median `sigma/d_nn` | sampled box mesh-ratio proxy | RMS `|m_0|` | RMS `|m_1|` | max local anisotropy |
|---|---:|---:|---:|---:|---:|---:|
| regular | 0.6 | 0.600 | 1.732 | 0.0746 | 0.143 | 1.416 |
| regular | 1.5 | 1.500 | 1.732 | 0.6000 | 0.193 | 1.737 |
| cluster/void | 0.6 | 1.362 | 4.401 | 1.2908 | 0.789 | 1.708 |
| cluster/void | 1.5 | 3.406 | 4.401 | 0.3810 | 0.277 | 2.829 |

The regular finite cloud’s zeroth-moment defect increased at large core
because a wider kernel extended beyond the truncated support.  The clustered
cloud’s median nearest-neighbour overlap looked “strong” even at
`sigma/ell=0.6`, while it could not reveal the void, and yet its coverage and
first-moment defects were the worst.  Increasing the core improved coverage
there while worsening directional conditioning.  This small but actual-kernel
example is enough to reject a one-number universal overlap rule; it is not a
spatial convergence experiment.  All rows are in
[`geometry.csv`](vpm-stability-theory/geometry.csv).

![Geometry and moment diagnostics](vpm-stability-theory/geometry_conditioning.png)

## 3. Finite-`N` smoothness and perturbation growth

Write the dimensionless kernel as

```text
K_sigma(r) = sigma^(-2) K_hat(r/sigma).
```

The exact Gaussian origin series makes `K_hat` smooth, while its derivatives
decay in the far field.  Hence, for every fixed derivative order `m`, a finite
constant `C_m` exists such that

```text
||D^m K_sigma(r)|| <= C_m sigma^(-(2+m)).                 (1)
```

For a sharper geometry-aware version let
`kappa_m(rho)=||D^m K_hat(rho)||` and define

```text
A_i^(m) = sum_j |Gamma_j| kappa_m(rho_ij)/sigma_ij^(2+m),
B_i^(m) = sum_j             kappa_m(rho_ij)/sigma_ij^(2+m).
```

Then `|U_i|<=A_i^(0)` and `||J_i||<=A_i^(1)`.  Differentiating
`S_i=J_i^T Gamma_i` gives the block structure

```text
D F = [ U_X       U_Gamma ]
      [ S_X       S_Gamma ],

partial_Xk U_i       = delta_ik sum_j DK_ij Gamma_j - DK_ik Gamma_k,
partial_Gammak U_i   = K_ik,
partial_Xk S_i       = (partial_Xk J_i)^T Gamma_i,
partial_Gammak S_i   = (partial_Gammak J_i)^T Gamma_i
                       + delta_ik J_i^T.                  (2)
```

For row-sum block bounds, the corresponding conservative estimates are

```text
||U_X||_row,i       <= 2 A_i^(1),
||U_Gamma||_row,i   <= B_i^(0),
||S_X||_row,i       <= 2 |Gamma_i| A_i^(2),
||S_Gamma||_row,i   <= A_i^(1) + |Gamma_i| B_i^(1).       (3)
```

These equations expose the small-core sensitivity: velocity, its gradient,
and its Hessian carry pair scales `sigma^-2`, `sigma^-3`, and `sigma^-4`.
Disorder enters through the actual `rho_ij` and strength-weighted sums; using
only `sigma_min` replaces those sums by a much more pessimistic worst case.

Positions and strengths have different units.  With reference scales `L` and
`G`, define the scaled block-maximum norm used for (3)–(4) by

```text
||(delta X,delta Gamma)||_(infinity,b) =
    max(max_i |delta X_i|/L, max_i |delta Gamma_i|/G).
```

A valid Lipschitz constant `L_(infinity,b)` in this norm is bounded by the
maximum of

```text
2 A_i^(1) + (G/L) B_i^(0),
(2L/G)|Gamma_i| A_i^(2) + A_i^(1) + |Gamma_i| B_i^(1).   (4)
```

Because all sums are finite when `sigma_min>0`, the exact-Gaussian `F_sigma`
is locally Lipschitz, proving unique local existence by the standard finite-
dimensional ODE theorem.  The same conclusion holds for the production
evaluator on any neighbourhood that does not intersect a pair surface
`|r_ij|/sigma_ij=0.2`; the unmatched branch values prevent a global local-
Lipschitz claim on that surface.  A bound of the form
`d||Gamma||/dt <= c||Gamma||^2` does not exclude finite-time growth, so smooth
regularization alone does not prove a global solution.

The logarithmic-norm/tangent calculation uses a different norm.  Let
`z=(X/L,Gamma/G)` and define the full scaled Euclidean norm
`||delta Y||_(2,s)=||delta z||_2`.  For `2N` three-vector blocks,

```text
||delta Y||_(infinity,b) <= ||delta Y||_(2,s)
                         <= sqrt(2N) ||delta Y||_(infinity,b).
```

Let `J_s(t)` be the Jacobian in these scaled Euclidean coordinates and let

```text
mu_2(J_s) = lambda_max((J_s+J_s^T)/2).
```

For the infinitesimal tangent `delta z'=J_s(t)delta z`, the exact trajectory
bound is

```text
||delta Y(t)||_(2,s) <= exp(integral_0^t mu_2(J_s(s)) ds)
                        ||delta Y(0)||_(2,s).              (5)
```

For a finite nonlinear difference between two trajectories, (5) requires
replacing the nominal-trajectory value by a supremum of `mu_2(J_s)` over a
tube/line segments connecting the trajectories; norm-equivalence factors are
needed if the result is restated in the block-maximum norm.  With an additive
residual, variation of constants adds the correspondingly weighted residual
integral.  The spectral abscissa of a frozen Jacobian is not a substitute for
(5): nonnormality can make `mu_2(J_s)-max Re(lambda(J_s))` positive, and `J_s`
changes with the particle trajectory.

## 4. What SSPRK3 can and cannot guarantee

Gottlieb, Shu, and Tadmor define strong-stability preservation relative to a
specified norm, seminorm, or convex functional for which forward Euler is
already strongly stable under `dt<=dt_FE`; the higher-order method inherits
that property through a convex-stage representation and a modified step
restriction (pp. 90–96 of the
[primary paper](https://math.umd.edu/~tadmor/pub/linear-stability/Gottlieb-Shu-Tadmor.SIREV-01.pdf)).
Here, stretching can produce real physical amplification, and no forward
Euler nonexpansiveness result has been proved.  Therefore “SSPRK3” establishes
the tableau and its conditional inheritance property, not VPM contractivity.

There is nevertheless a rigorous, if crude, finite-nonlinear-difference bound.
Using the scaled block-maximum norm above and a Lipschitz constant valid on a
tube containing both stage trajectories, the three SSPRK stages give

```text
P3(z) = 1 + z + z^2/2 + z^3/6,
||Phi_coupled(Y)-Phi_coupled(Z)||_(infinity,b)
    <= P3(dt L_F)||Y-Z||_(infinity,b).
                                                               (6)
```

Split `F=A+B` with `A=(U,0)` and `B=(0,S)`.  The actual historical sequence
is `A` SSPRK3 for a full step and then `B` SSPRK3 for a full step, so

```text
||Phi_historical(Y)-Phi_historical(Z)||_(infinity,b)
    <= P3(dt L_B) P3(dt L_A) ||Y-Z||_(infinity,b).             (7)
```

Because `L_F<=L_A+L_B` and, for nonnegative `a,b`,
`P3(a+b)<=P3(a)P3(b)`, the coupled *upper bound* is no worse than the
historical product bound in the same norm.  Both bounds normally exceed one
and can be very pessimistic; this is not a nonlinear stability theorem.

A Taylor expansion of the exact subflows gives the historical local split
defect

```text
(dt^2/2) C_AB(Y) + O(dt^3),
C_AB = B'(Y)A(Y) - A'(Y)B(Y)
     = (-U_Gamma S, S_X U).                                   (8)
```

The two components contain leading pair scales as severe as `sigma^-5` and
`sigma^-6` before cancellations and dimensional scaling.  Thus the split
error constant can worsen rapidly with small cores and clustered/high
strengths even though the regularized ODE remains smooth.  Replacing the
sequence by `A/2`-`B`-`A/2` cancels (8), producing a second-order composition.
Using SSPRK3 for each subproblem does not make that composition third order;
its nine defined subproblem calls per step retain a second-order splitting
defect.  The common-stage SSPRK3 applies a third-order RK method directly to
`A+B`, uses three coupled calls per step, and has no split commutator.  These
call counts describe the method definitions, not a wall-time ratio: the probe
computes velocity and gradient together even when one block is zeroed, while
production split paths may perform cheaper specialized work.

## 5. Numerical accuracy and tangent evidence

The temporal study integrated a smooth, analytically solenoidal sampled field
to a fixed horizon, with a 512-step coupled RK4 reference.  The reported error
is the RMS of the full scaled Euclidean state difference: positions are scaled
by `ell=1`, strengths by the initial RMS particle strength, and the concatenated
Euclidean norm is divided by `sqrt(6N)`.  This is not the block-maximum norm
used for (3)–(4).  The method maps make 3, 6, and 9 defined calls per step for
coupled, historical, and symmetric updates.  Because the probe evaluates both
velocity and gradient on every call before zeroing an unused block, these
counts must not be read as production cost ratios.

| Method | Finest-pair order across 12 cases | Median error at `dt=0.00625` | Error range at `dt=0.00625` | defined calls/step |
|---|---:|---:|---:|---:|
| coupled SSPRK3 | 2.974–3.000 (median 3.000) | `1.83e-13` | `3.39e-15–1.81e-11` | 3 |
| historical `A` then `B` | 0.99993–1.00010 (median 1.000) | `2.57e-6` | `3.31e-7–2.17e-5` | 6 |
| symmetric `A/2`,`B`,`A/2` | 1.99995–2.00007 (median 2.000) | `2.81e-10` | `1.69e-11–7.19e-9` | 9 |

At the coarsest tested `dt=0.05`, representative scaled errors were:

| Cloud, `sigma/ell` | coupled | historical | symmetric |
|---|---:|---:|---:|
| regular, 0.6 | `1.17e-9` | `6.83e-5` | `8.91e-8` |
| regular, 1.5 | `3.19e-12` | `3.48e-6` | `1.50e-9` |
| cluster/void, 0.6 | `9.25e-9` | `1.74e-4` | `4.60e-7` |
| cluster/void, 1.5 | `1.73e-12` | `2.65e-6` | `1.08e-9` |

The result is a temporal refinement result on the defined ODE, not evidence
that small-core solutions are spatially more accurate.  Smaller cores made
the temporal constants larger in these probes, as the derivative scaling
predicts.  Several finest coupled errors are within a few multiples of binary64
roundoff and the refined-reference floor (minimum `3.39e-15`); absolute method
ratios at those rows are not meaningful.  The order conclusion is supported
by the resolved coarser refinements as well as the reported finest-pair range.

![Temporal refinement](vpm-stability-theory/temporal_error.png)

The `N=8` tangent study used six steps of `dt=0.05`, formed centered-difference
Jacobian estimates for every numerical step in full scaled Euclidean
coordinates, multiplied the time-dependent step Jacobians, and compared the
largest singular-value estimate with a refined RK4 flow-map finite-difference
estimate.  Five nominal-trajectory samples supplied a trapezoidal
logarithmic-norm diagnostic and frozen-Jacobian spectral-abscissa samples.
Neither finite differences nor five-point quadrature certify the exact tangent
or the continuous integral in (5).

| Cloud | `sigma/ell` | refined tangent `s_max` estimate | sampled `exp(trapz mu_2)` | coupled/ref estimate | historical/ref estimate | symmetric/ref estimate | sampled max nonnormal gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| regular | 0.6 | 1.013839 | 1.013839 | 0.9999999998 | 1.0000043235 | 0.9999999999 | 0.01747 |
| regular | 1.5 | 1.005780 | 1.005780 | 0.9999999998 | 0.9999997468 | 0.9999999998 | 0.00636 |
| cluster/void | 0.6 | 1.039544 | 1.039548 | 0.9999999991 | 1.0000556918 | 1.0000000555 | 0.05255 |
| cluster/void | 1.5 | 1.006511 | 1.006511 | 1.0000000000 | 1.0000000736 | 1.0000000001 | 0.01227 |

The estimates indicate that the physical flow tangent itself grows
(`s_max>1`), and the sampled positive nonnormal gap demonstrates why
eigenvalues of one frozen Jacobian are insufficient.  Coupled SSPRK3 showed no
resolved extra amplification at this step; the historical split’s largest
estimated excess, `5.57e-5`, occurred in the clustered underlapped case.  This
is an infinitesimal, short-horizon numerical estimate—not a finite nonlinear
difference bound or a guarantee at larger `dt` or longer times.  Step-product
and directly differenced full-map estimates agreed within `1.1e-9`
relatively.  See
[`tangent.csv`](vpm-stability-theory/tangent.csv).

![Tangent amplification relative to the refined flow](vpm-stability-theory/tangent_growth.png)

## 6. A defensible diagnostic hierarchy

No single universal stability condition emerged.  The following hierarchy is
both checkable and honest about what each level controls.

1. **Finite-ODE admissibility:** require finite state, fixed `N` over the
   analyzed substep, and `sigma_min>0`; for the literal current evaluator,
   also monitor/correct the `rho=0.2` splice before invoking a globally smooth
   RHS theorem.  On the intended time interval, bound
   strengths and the geometry-aware sums `A_i^(1)`, `A_i^(2)`, `B_i^(0)`, and
   `B_i^(1)`.  This supplies local existence and a computable Lipschitz/log-
   norm perturbation estimate.
2. **Spatial sampling:** report distributions—not only medians—of
   `sigma/ell_i` and `sigma/d_i`, plus a stated sampled fill/mesh-ratio proxy,
   coverage, first-moment defect, and directional conditioning on a stated
   support.  If invoking Cottet’s convergence theorem, check its actual
   increasing-overlap and kernel-moment assumptions; do not replace them by
   an informal constant-overlap rule.
3. **Time-step perturbation screen:** for a chosen allowed per-step
   amplification `K>1`, (6) is sufficient if `P3(dt L_F)<=K`; (7) is the
   corresponding historical bound.  A rigorously bounded continuous
   `mu_2` integral can sharpen it.  A sampled integral or finite-difference
   tangent map is only a diagnostic estimate and does not bound finite
   nonlinear differences without a trajectory-tube argument.
4. **Temporal accuracy:** perform reference refinement with the same spatial
   state and operators.  Verify third-order decay for the coupled method
   before treating a time step as resolved.  The 3/6/9 defined call counts in
   this probe establish the compositions, not production wall time; an actual
   cost comparison needs timed specialized kernels or counted pair operations.
5. **Complete-solver qualification:** separately test core/diffusion
   composition, external forcing stage consistency, induction approximation,
   particle lifecycle maps, and health/rejection logic.  Fixed-ODE evidence
   cannot certify those maps.

This hierarchy is a stability/accuracy framework, not a new magic CFL number.
The most useful operational fallback is demonstrated temporal error reduction
combined with tracked geometry/moment diagnostics.

## 7. Transfer to the old-wake rotor diagnosis

The fixed-core study can inform the planned rotor reruns, but only as a set of
measurements and discriminating experiments.

### Measurements worth transferring

- Stratify the wake by age and downstream distance, then record distributions
  of `sigma_i/V_i^(1/3)`, `sigma_i/d_i`, nearest-neighbour distance, and a local
  separation/fill or void proxy.  A global median will hide an old-wake void
  next to a cluster.
- On deterministic representative subsamples, compute Gaussian coverage,
  zeroth/first moment defects, and local moment-matrix conditioning.  Keep the
  finite support/boundary convention explicit.
- Track `|Gamma_i|`, `||grad u_i||`, strain, and geometry-aware derivative-sum
  proxies analogous to `A_i^(1)` and `|Gamma_i|A_i^(2)`.  Their correlation
  with wake age identifies where the fixed-core tangent/commutator constants
  become large.
- A rotor time-step study is not automatically a time-only refinement: current
  `dt`-dependent shedding changes wake insertion and particle count/state
  dimension.  Use it as a temporal convergence test only if supported controls
  can hold shedding locations/circulation, core/LES rules, lifecycle decisions,
  and spatial resolution equivalent.  Otherwise use a supported frozen-
  snapshot/replay subproblem or label the run a joint space-time study.  Do not
  infer third-order time behavior from a sequence whose particle discretization
  changes with `dt`.
- Where affordable, evolve a small paired perturbation or compute sampled
  tangent/log-norm diagnostics on frozen wake snapshots.  Compare amplification
  with the underlying refined flow, not with unity, because physical
  stretching can legitimately grow perturbations.

### Interpretive use

- Growth that decreases at the coupled method’s refinement rate is temporal
  error evidence.
- Growth that is time-step insensitive but co-locates with poor coverage,
  large mesh ratio, anisotropy, or derivative sums points toward spatial
  representation/conditioning or physical stretching, not an SSP failure.
- A historical split comparison may expose the commutator mechanism, but it
  should not be reinstated as a production remedy: it is generically first
  order.  The present evaluation counts do not establish a production wall-
  time disadvantage.

### Conclusions that do not transfer

No numerical value from the `N=8/27` probes is a rotor acceptance threshold.
The tangent ratios do not include a driven VLM boundary, wake-particle
creation, changing `N`, core spreading, viscosity, LES eddy viscosity,
relaxation, redistribution/remeshing, tree/FMM approximation, moving bodies,
or accepted-step lifecycle logic.  Core evolution changes the vector field
whose derivatives were bounded; insertion/deletion changes the state space;
VLM forcing makes the system nonautonomous and may itself be lagged or
stage-coupled.  The old-wake problem therefore requires its own rotor evidence.
This report supplies measurement definitions and falsifiable temporal tests,
not a rotor stability conclusion.

## 8. Exact status: proved, observed, unresolved

### Proved under stated assumptions

- Smooth local well-posedness of the finite fixed-core *exact-Gaussian* ODE for
  `sigma_min>0`, and piecewise-smooth local well-posedness of the production
  evaluator away from `rho=0.2` pair surfaces.
- Kernel derivative scaling (1), block structure (2), and conservative bounds
  (3)–(4).
- Scaled-Euclidean infinitesimal logarithmic-norm inequality (5), with the
  separate tube-supremum requirement for finite nonlinear differences.
- SSPRK3 Lipschitz polynomial bounds (6)–(7), while explicitly not claiming
  contractivity.
- The historical commutator defect (8), generic global orders one/two/three
  for historical/symmetric/coupled methods under standard smoothness.

### Observed in bounded actual-kernel probes

- Analytic-gradient/finite-difference agreement including the regularized
  origin.
- Orders one, two, and three in every resolved cloud/core refinement.
- Smaller-core and clustered cases increased temporal error constants and
  physical/nonnormal tangent growth.
- At the tested step, the coupled centered-difference tangent estimate matched
  the refined-flow estimate to numerical differentiation resolution and the
  coupled method materially outperformed both split alternatives in temporal
  error where errors were above the floating-point/reference floor.

### Unresolved or expressly not claimed

- A VPM-specific SSP/contractivity CFL theorem.
- A globally smooth-ODE theorem for the literal unmatched production kernel
  splice at `rho=0.2`.
- Global-in-time bounded strengths for arbitrary finite clouds.
- A universal overlap/disorder threshold.
- Full spatial convergence of the production Gaussian method on irregular,
  adaptive finite wakes.
- Stability of the complete accepted-step solver or the driven rotor wake.

## 9. Reproducibility and sources

The complete machine-readable output is
[`results.json`](vpm-stability-theory/results.json); the script and exact run
command are in [`README.md`](vpm-stability-theory/README.md).  The final run
used one CPU thread, 25.38 s wall time as measured inside the script, and
115,818,496 bytes maximum RSS.  It used no GPU and installed no packages.

Primary-source verification, including inaccessible/full-text distinctions,
is recorded in
[`theorem-source-ledger.md`](vpm-stability-theory/theorem-source-ledger.md).
The principal sources are Cottet’s
[1988 paper](https://ems.press/content/serial-article-files/17460),
Winckelmans’ [1989 thesis](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf),
and Gottlieb, Shu, and Tadmor’s
[2001 SSP paper](https://math.umd.edu/~tadmor/pub/linear-stability/Gottlieb-Shu-Tadmor.SIREV-01.pdf).

## 10. Handoff and resource release

Boss accepted the corrected result only in its stated scope: bounded temporal-
accuracy evidence and conditional perturbation analysis for the fixed-core
ODE, not a general full-VPM or rotor stability theorem.  Boss owns the separate
kernel-splice issue and has sent the accepted limited finding onward to Writer;
no rotor causality is asserted.

Across all probe attempts, including two interrupted/failed setup passes, the
visible process/script timers total approximately 338 s (5 min 38 s), below
the 10 min cumulative budget.  The highest reported per-process maximum RSS
was 131,235,840 bytes (about 125 MiB), below 512 MiB.  Every numerical run used
one CPU thread; no GPU or package installation was used.  All process sessions
and study-owned temporary download/render caches were released after final QA.
