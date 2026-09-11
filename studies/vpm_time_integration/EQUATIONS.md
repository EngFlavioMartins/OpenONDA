# Equation ledger

This is the compact, package-local map. The complete derivations and source
anchors remain in the immutable historical records:
[`fixed_core_review.md`](provenance/reviews/fixed_core_review.md) and
[`fixed_core_equation_map.md`](provenance/reviews/fixed_core_equation_map.md).

## Fixed-core particle ODE

For `r_ij=X_i-X_j`, `sigma_ij=(sigma_i+sigma_j)/2`, and
`rho_ij=|r_ij|/sigma_ij`,

```text
zeta(rho) = pi^(-3/2) exp(-rho^2),
q(rho)    = [erf(rho) - (2/sqrt(pi)) rho exp(-rho^2)]/(4 pi),
U_i       = sum_j q(rho_ij) (Gamma_j x r_ij)/|r_ij|^3,
J_i       = grad U(X_i),
S_i       = J_i^T Gamma_i,
Y'        = F(Y) = (U(X,Gamma), S(X,Gamma)).
```

The exact Gaussian expansion `q(rho)=rho^3/(3 pi^(3/2))+O(rho^5)` gives a
finite analytic origin extension. For fixed finite `N` and pair cores bounded
below by `sigma_min>0`, the exact-Gaussian finite-particle vector field is
smooth and locally Lipschitz. This is a local result; quadratic stretching does
not supply global-in-time strength boundedness.

## Method decomposition and order

Split `F=A+B`, with `A=(U,0)` and `B=(0,S)`. For the historical full `A` step
followed by a full `B` step, Taylor expansion gives

```text
Phi_B(dt) Phi_A(dt) - Phi_(A+B)(dt)
    = (dt^2/2) [B'(Y)A(Y) - A'(Y)B(Y)] + O(dt^3).
```

The nonzero commutator makes this composition first order globally in the
generic smooth case. The symmetric `A/2-B-A/2` composition cancels that leading
term and is second order; applying SSPRK3 directly to `A+B` is third order.

For an RHS Lipschitz constant `L_F` valid on a tube containing both stage
trajectories in the scaled block-maximum norm,

```text
P3(z) = 1 + z + z^2/2 + z^3/6,
||Phi_coupled(Y)-Phi_coupled(Z)|| <= P3(dt L_F) ||Y-Z||.
```

This is a finite-step perturbation-growth upper bound, not contractivity. The
sequential product has the corresponding conservative factor
`P3(dt L_B) P3(dt L_A)`.

## Geometry diagnostics

For volume-based particle spacing `h_i=V_i^(1/3)`, nearest-neighbour distance
`d_i`, sampled fill distance `h_X`, and separation radius `q_X`, the study
records `sigma_i/h_i`, `sigma_i/d_i`, and sampled `h_X/q_X`. At a diagnostic point
`x`,

```text
w_j(x) = V_j sigma_j^(-3) zeta(|x-X_j|/sigma_j),
m_0(x) = sum_j w_j(x) - 1,
m_1(x) = sum_j w_j(x) (X_j-x)/sigma_j.
```

These complementary spatial diagnostics do not define a universal
time-integration stability threshold.

## Norm separation

The conservative Jacobian row-sum theory uses a scaled maximum over
per-particle Euclidean position/strength blocks. Temporal errors, tangent
singular values, and logarithmic norm `mu_2` use full scaled Euclidean
coordinates. The two norms are related for `2N` three-vector blocks but are not
interchanged in the claims.

## Linear counterexample

For the oscillator with `q=omega dt`, coupled SSPRK3 has

```text
R(z)=1+z+z^2/2+z^3/6,
|R(iq)|^2=1-q^4/12+q^6/36.
```

Its imaginary-axis boundary is `q=sqrt(3)`. Exact-subflow Lie and symmetric
splits have unit-determinant elliptic eigenvalues for `0<q<2`, but do not
preserve the physical Euclidean norm and lose bounded powers at the boundary.
This separate example shows why order evidence is not a universal stability
ordering.
