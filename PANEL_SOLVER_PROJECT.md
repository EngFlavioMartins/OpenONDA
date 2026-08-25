# Panel Solver Reliability and Coupling Project

## Objective

Make the VPM panel solver a reliable, scalable body-boundary model for coupled
FVM–VPM simulations and for future complex geometries. The completed work must
support clean, complicated closed STL bodies without silently accepting invalid
geometry, and it must apply one consistent body-induced velocity to every
consumer of the VPM field.

The current cube investigation is documented in
`COUPLER_INVESTIGATION_LOG.md`. Preserve its production baseline until a
controlled same-seed experiment proves an improvement.

## Non-negotiable constraints

- Do not change the numerical formulas or behavior of `vorticity_mixed`.
- Do not add, restore, or call remeshing outside the VPM's existing GBD step.
- Do not modify the FVM-to-VPM absolute injection algorithm.
- Preserve `Gamma = cell_volume * FVM_vorticity` and hard replacement as the
  production baseline.
- A fixed-time panel refresh must never advance panel history, kinematics,
  forces, time counters, or wake shedding.
- Do not retain deprecated modes or dead compatibility code. Rejected
  experiments must be removed and their settings restored.
- Record every controlled experiment and disposition in
  `COUPLER_INVESTIGATION_LOG.md`.

## Required feature requests

### P0 — Correct and explicit coupling semantics

- [x] Define and document each `coupling_scope` in one authoritative location.
      See the `PanelSolver` class docstring in
      `source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py`.
- [x] Under `full`, solve panel strengths using **all active particles** and add
      panel-induced velocity to **all active particle trajectories** at every
      Runge–Kutta stage. Never distinguish injected from retained particles.
      Traced and confirmed already true of the existing implementation (see
      `COUPLER_INVESTIGATION_LOG.md`, 2026-08-25); not new code.
- [x] Make the existing coupler refresh calls operate for synchronized `full`
      coupling as well as `vpm_boundary_condition`: once after VPM evolution
      before the FVM boundary trace and once after FVM particle replacement.
- [x] Keep the refresh side-effect free. Unit-test panel history, step counters,
      kinematics, forces, and shedding before and after it.
      `tests/vpm/test_boundary_element_state_refresh.py`.
- [ ] Ensure sampling, pressure evaluation, velocity gradients, FVM boundary
      evaluation, and particle advection use the same current panel solution.
      The refresh now fires for `full` as well as `vpm_boundary_condition`,
      which closes the replaced-particle-state gap; a positive end-to-end
      test that every listed consumer reads the same-time solution has not
      been written.
- [ ] Add diagnostics for panel solve success/residual, panel refresh count,
      maximum/RMS panel-induced velocity at particles and FVM boundary faces,
      no-penetration residual, and timing by stage.
      Solve success, relative residual, iteration count, refresh count, and
      scheduled/subsampled max&RMS induced velocity at particles and FVM
      boundary faces are recorded. No-penetration residual and per-stage
      timing are not implemented (see `COUPLER_INVESTIGATION_LOG.md`).

### P0 — STL preflight and fail-fast validation

- [x] Add a public mesh-audit API/CLI for ASCII and binary STL files.
      `source/solvers/vpm/boundary_elements/panels/geometry/stl_audit.py`
      (`audit_stl_mesh`) and `scripts/panel_mesh_audit.py`.
- [x] Reject non-finite coordinates, zero-area/duplicate triangles, open edges,
      non-manifold edges, inconsistent winding, disconnected components that
      were not declared as separate bodies, and panel-count overflow.
- [ ] Report poor aspect ratios, extreme scale, close/overlapping components,
      and likely self-intersections. Do not silently repair geometry.
      Aspect ratio is reported, and an approximate proximity heuristic warns
      on close/overlapping/self-intersecting triangle pairs
      (`test_overlapping_components_are_reported_not_rejected`). Extreme-scale
      reporting and an exact intersection test are not implemented.
- [x] Orient each closed connected component using topology and signed volume;
      remove the geometric-centre normal heuristic for general concave bodies.
- [x] Produce a machine-readable audit report containing counts, bounding box,
      signed volume, area/aspect-ratio statistics, connectivity, and disposition.
- [ ] Support multiple closed bodies explicitly. State and enforce the policy
      for cavities and nested components.
      The stated and enforced policy is currently restrictive:
      `PanelSolver.add_surface` accepts exactly one connected component and
      rejects multi-component STLs, because one file maps to one `PanelBody`
      with one uid/kinematics. Genuine multi-body support (one body per
      component, independently identified and moved) and a cavity/nested
      orientation policy are not implemented; `audit_stl_mesh` can report
      multiple components but nothing consumes that yet.

### P1 — Complex-geometry and performance support

- [ ] Permit a dedicated coarse panel STL separate from the detailed FVM STL,
      with file hash and preprocessing provenance in run metadata.
- [ ] Provide an opt-in decimation/preprocessing command with target error and
      panel-count controls. Never decimate automatically during a solver run.
- [ ] Estimate dense-matrix memory, assembly cost, and panel–target interaction
      cost before allocation; fail with an actionable message if unsafe.
- [ ] Reuse the influence matrix/factorization for static geometry.
- [ ] Add a robust preconditioned iterative solve, convergence history, and a
      hard failure when the requested residual is not reached.
      The hard failure is implemented: both solvers now decide success from
      the relative residual of the returned solution (recomputed after the
      NEUMANN gauge projection), and `PanelSolver` raises by default. No
      preconditioner and no per-iteration convergence history yet.
- [ ] Establish a scalable panel-to-particle evaluation path so complicated
      geometries do not require an unbounded direct
      `N_panels * N_particles` calculation.
- [ ] Make float32/float64 behavior explicit and test both precisions.
- [ ] Replace brute-force inside/collision queries with a robust accelerated
      method suitable for concave and multi-body meshes.

### P1 — Numerical verification suite

- [ ] Sphere: verify analytic potential-flow velocity, far-field decay, surface
      no-penetration, and mesh convergence.
      Only the far-field decay *exponent* is verified
      (`tests/vpm/test_panel_solver_sphere_analytic.py`), plus a tight solve
      residual. The decay magnitude and sign, the analytic surface velocity,
      no-penetration, and true error-vs-refinement convergence are all still
      unverified (see `COUPLER_INVESTIGATION_LOG.md`, 2026-08-25).
- [ ] Ellipsoid: verify convergence against an independent reference.
- [ ] Concave watertight body: verify orientation, finite induced velocity,
      solver convergence, and no-penetration without relying on a centroid rule.
- [ ] Multiple close bodies: verify component orientation, mutual influence,
      conditioning diagnostics, and deterministic results.
- [x] Invalid-STL corpus: holes, flipped faces, duplicate/degenerate faces,
      non-manifold edges, disconnected components, and self-intersections must
      fail or warn exactly as documented.
      `tests/vpm/test_panel_stl_audit.py` covers each listed case: open edges,
      a reversed face (inconsistent winding), duplicate and zero-area faces,
      a non-manifold edge, disconnected components, and an overlapping-pair
      warning. The self-intersection case is a proximity heuristic, not an
      exact intersection test.
- [ ] Lifecycle integration test: prove that synchronized `full` coupling uses
      all particles during advection and the same-time state for every boundary
      and sampler evaluation.

## Controlled cube-flow decision test

After the P0 lifecycle fix, restart both candidates from the preserved exact
`t=2` checkpoint and run to `t=2.5` sequentially:

1. Production `vpm_boundary_condition` baseline.
2. Synchronized `full` panel coupling, with no other change.

Compare drag, FVM/VPM centreline and off-axis profiles, panel-induced velocity,
normal-velocity residual, median coupled/VPM step time, and pre-replacement
particle count. Use the fresh fully meshed FVM reference and the universal 5%
gate. Adopt `full` only if controlled evidence shows a material accuracy
improvement without instability or unacceptable cost; otherwise restore the
production baseline. Do not use this test to tune any unrelated parameter.

## Acceptance criteria

- All geometry, solver, lifecycle, and restart tests pass deterministically.
- Analytic/reference benchmarks demonstrate convergence under panel refinement.
- Surface no-penetration residuals and linear-solver residuals are reported and
  meet documented tolerances; non-convergence is fatal.
- Complex-STL preprocessing and solver resource requirements are bounded and
  reported before execution.
- Cube-flow disposition is supported by the same-seed evidence above.
- Existing focused coupler/VPM tests, Ruff, and Pyrefly pass.
- No production change is retained without an entry in
  `COUPLER_INVESTIGATION_LOG.md` identifying the problem, controlled evidence,
  result, and rollback disposition.

## Deliverables

- Production implementation with obsolete/rejected code removed.
- Focused unit, integration, analytic, invalid-geometry, restart, and performance
  tests.
- User documentation for STL requirements, coupling scopes, diagnostics,
  supported scale, preprocessing, and failure recovery.
- Reproducible benchmark artifacts and cube-flow comparison figures/tables.
- Updated investigation ledger and one or more scoped Git commits.

## Explicitly out of scope

- FVM–VPM strength/deconvolution correction.
- Changes to `vorticity_mixed`.
- New remeshing or regeneration behavior.
- Calibration of eta, GBD pruning, transfer bounds, core radius, or time step.
