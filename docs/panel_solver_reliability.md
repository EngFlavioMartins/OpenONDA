# Panel solver reliability contract

The moving multi-body Neumann boundary solver is production qualified within
the domain stated in [panel_solver_qualification.md](panel_solver_qualification.md).
That page owns the numerical evidence, limits, scaling data, and capability
matrix. This page defines the runtime contract.

## Physics and sign convention

At collocation point `i`, the supported Neumann formulation enforces

```text
(U_inf + u_incident + u_panel - u_body) . n_i = 0
A sigma = -(U_inf + u_incident - u_body) . n
```

- `freestream_velocity` passed to `solve` is `U_inf`.
- `lattice.incident_velocity` is VPM/external velocity only.
- `lattice.body_velocity` is rigid velocity
  `V + omega x (x - (rotation_centre + translation))`.
- `source_strength` is the constant source density `sigma` on each triangle;
  positive strength is outward flux with the oriented outward normal.
- `surface_velocity_absolute` is `U_inf + u_incident + u_panel`.
- `surface_velocity_relative` subtracts `u_body` and is the field used for
  impermeability diagnostics.

Each body has one authoritative `BodyPose`. Geometry is evaluated once as
`R @ (x0 - c) + c + T`; kinematics compose pose state rather than applying
sequential coordinate mutations.

## Per-body compatibility and linear solvers

Every disconnected closed body imposes one area-weighted source constraint:

```text
sum(sigma_i * area_i) = 0
```

The solver minimizes `||A sigma - b||` subject to all body constraints. It
does not modify an unconstrained solution after the fact.

- `linear_solver="SCIPY"` uses a reusable null-space/pivoted-QR factorization.
  Geometry revisions invalidate the complete AIC and factors; changing only
  the incident field reuses them.
- `linear_solver="BICGSTAB_GPU"` uses projected CGLS for Neumann problems.
  Every gradient and search direction is projected into the per-body feasible
  subspace. For standalone Dirichlet it retains unconstrained BiCGSTAB.

For constrained Neumann solves, `||A sigma-b||/||b||` is reported as
`discrete_equation_residual`; it is a finite-resolution compatibility metric,
not the convergence test. Success requires both relative flux and projected
KKT stationarity to meet `residual_tolerance`.

## Geometry and STL requirements

Each `add_surface` call accepts one finite, closed, watertight,
consistently-wound triangulated component with no zero-area or duplicate
triangles and no open or non-manifold edges. Orientation is corrected from
topological signed volume before allocation.

One STL maps to one independently identifiable and movable body. A
multi-component STL is rejected; add separate files as separate bodies.
Nested/cavity-shell orientation and exact triangle-triangle self-intersection
testing remain outside the supported geometry contract.

Use `python scripts/panel_mesh_audit.py <file.stl> --strict` before production
runs. Disabling validation is a debugging-only path.

## Coupling and load capabilities

Neumann VPM coupling consumes the current particle-induced velocity as
`incident_velocity`, solves the harmonic source correction, and can add that
correction to every active particle on device. `refresh_coupled_solution`
updates this state without advancing kinematics or history.

The coupling scopes `full`, `vpm_boundary_condition`, `normal`, and `pressure`
are defined by the `PanelSolver` class docstring. Their load limitation is
common: static steady-potential pressure/loads are qualified; moving-body and
general vortical VPM-coupled loads are not. Moving load APIs raise instead of
silently applying incomplete steady Bernoulli physics.

VPM plus Dirichlet is rejected because a general vortical incident field does
not provide the scalar potential required by that formulation. Standalone
Dirichlet remains experimental and is not on the production path.

## Far field and supported scale

Per-body far-field moments retain monopole and dipole terms. A target uses the
expansion only when its distance exceeds `far_field_acceptance * body_radius`
and the total panel threshold is met. Qualification selected the default
acceptance `5.0`; below `far_field_min_panels=256`, evaluation remains exact.

The AIC is dense. CPU factor reuse retains roughly four dense arrays (AIC,
null-space basis, Q, and R), and the memory guard budgets them before
allocation. Projected CGLS retains only the AIC plus O(N) work vectors. The
qualification campaign covers 256–2,000 panels across 1–8 bodies and a
4,000-panel extension; it does not claim asymptotically scalable dense BEM.

The tested near-contact domain is `g/h >= 0.5`. Smaller gaps require new
resolution evidence before use.

## Diagnostics and failure recovery

`results["diagnostic_history"]` records the actual solver, requested strategy,
equation/constraint/KKT metrics, wall RMS/max, per-body net flux, factor reuse,
AIC/cache bytes, iterations, and optional synchronized stage timings.

- Solve failure: inspect projected optimality, relative flux, wall residual,
  and the STL audit separately. Do not interpret a nonzero constrained
  equation residual as iterative non-convergence.
- Memory failure: lower `max_n_panels`, select projected CGLS, or raise the
  budget only when the hardware can hold the stated allocation.
- Near-contact use below `g/h=0.5`: refine or stop; this is outside the
  qualified envelope.
- Moving/load exception: no supported unsteady force model is present; do not
  bypass the exception and relabel steady pressure as unsteady pressure.
