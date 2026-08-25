# Panel solver: STL requirements, coupling scopes, and diagnostics

This page is an index into the authoritative sources for each topic, not a
duplicate of them — see `PANEL_SOLVER_PROJECT.md` for the reliability program
this work belongs to, and `COUPLER_INVESTIGATION_LOG.md` for the evidence and
disposition behind each change.

## STL requirements

A body STL loaded through `PanelSolver.add_surface` must be a closed,
watertight, consistently-wound triangulation with exactly one connected
component: finite coordinates, no zero-area or duplicate triangles, and no
open or non-manifold edges. Normal orientation is corrected automatically
from the component's topological signed volume — correct for concave bodies,
unlike a per-panel centroid test.

The STL is loaded, audited, and oriented **before** any Taichi lattice or
dense influence matrix is allocated, so invalid geometry is rejected without
consuming GPU memory.

Multi-component STLs are **rejected**, not merged: `add_surface` maps one
file to one `PanelBody` with one uid and one kinematics object, so separate
shells would be silently fused and could not be identified or moved
independently, and nested shells would need a cavity-orientation policy that
does not exist yet. Add each body from its own STL. (`audit_stl_mesh` can
report multiple components for inspection, but no solver path consumes that.)

- Full check list, tolerances, and the machine-readable report schema:
  `source/solvers/vpm/boundary_elements/panels/geometry/stl_audit.py`
  (`audit_stl_mesh` docstring).
- Command-line audit: `python scripts/panel_mesh_audit.py <file.stl>
  [--max-panels N] [--expected-components N] [--json report.json] [--strict]`.
- Set `validate=False` on `add_surface`/`add_body_from_mesh_stl` only to
  reproduce pre-audit behavior for debugging; production code should not
  disable it.

## Coupling scopes

`PanelSolver(coupling_scope=...)` is the single switch controlling how a
panel body participates in a VPM step. The authoritative definition of
`"full"`, `"vpm_boundary_condition"`, `"normal"`, and `"pressure"` — what each
one solves, injects, and gets refreshed by a coupler — is the `PanelSolver`
class docstring in
`source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py`. Do not
duplicate that definition elsewhere; update it there and this page stays
correct by reference.

## Diagnostics

`PanelSolver.results["diagnostic_history"]` accumulates one entry per solve
with `residual` (relative, recomputed from the strengths actually left in the
lattice after any NEUMANN gauge projection), `iterations`,
`linear_solver_success`, and `refresh_count`.

A solve counts as converged only when that relative residual is at or below
the solver's `residual_tolerance` (`1e-8` by default); a non-finite solution
is treated as infinite residual. This matters because a dense direct solve
returns a vector for a rank-deficient or inconsistent system without raising.
Non-convergence raises by default (`raise_on_non_convergence=True`); pass
`False` only for a caller that checks `linear_solver_success` itself.

**Panel-induced-velocity diagnostics are off by default.** Evaluating every
panel at every particle is an `n_panels * n_particles` direct calculation —
tens of millions of interactions per refresh for a coupled cube run, twice
per coupling step. Set `diagnostic_interval_steps > 0` to enable them; they
then run on that refresh schedule and read a deterministic fixed-stride
subsample bounded by `diagnostic_sample_size` (4096 by default), recording
`max_induced_velocity_at_particles`, `rms_induced_velocity_at_particles`,
`induced_velocity_sample_size`, and `induced_velocity_sample_stride`. The
same schedule gates the coupler's `[Coupler][BoundaryPanelVelocity]` log
line, which otherwise repeats work the boundary trace already performs.

## Supported scale and preprocessing

`PanelSolver(memory_budget_bytes=...)` fails fast, before any GPU allocation,
if the dense `max_n_panels x max_n_panels` influence matrix would exceed the
budget (default 4 GiB), naming the panel count that would fit. There is no
coarse/detailed dual-STL support, decimation tooling, accelerated
panel-to-particle evaluation, or genuine multi-body support yet — see the
"Deferred" list in the 2026-08-25 panel-solver entry of
`COUPLER_INVESTIGATION_LOG.md` for the full set of open P1 work and why each
item was not started in this pass.

## Failure recovery

- STL audit failure: `scripts/panel_mesh_audit.py <file.stl> --json report.json`
  names the specific failing check and offending count; fix the mesh, or split
  a multi-component file into one STL per body, rather than disabling
  `validate`.
- Panel solve failure (`RuntimeError` from `PanelSolver.solve`): the message
  includes the panel count and the achieved relative residual against the
  tolerance. It means the requested tolerance was not reached, not that the
  geometry is necessarily invalid — run the STL audit first to rule that out.
  A near-`1.0` relative residual usually indicates a rank-deficient or
  inconsistent system (degenerate geometry or a duplicated body).
- Memory-budget failure: lower `max_n_panels` to the suggested value, or
  raise `memory_budget_bytes` only if that much memory is actually available.
