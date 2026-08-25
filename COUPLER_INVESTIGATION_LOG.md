# FVM–VPM Coupler Investigation Ledger

Last updated: 2026-08-25

This file is the durable record of coupling experiments. A setting is retained
only when a controlled comparison improved the result or fixed a demonstrated
contract violation. Rejected settings are restored to the stated baseline.

## Current implementation — full-solver acceptance pending

The production transfer path is now an absolute common-lattice state blend:

1. Map fluid-cell circulation `Gamma_c = V_c*omega_c` to the regular VPM
   lattice with complete M4' support.
2. Map the replaceable current VPM circulation to the same lattice and time.
3. Apply `Gamma_new = eta*Gamma_FVM + (1-eta)*Gamma_VPM` node by node in a C1
   overlap band. Remove the VPM source particles represented by this state,
   insert the blended lattice state once, and leave VPM-authoritative particles
   untouched. Coincident release nodes are merged rather than duplicated.
4. Remove the blend-only cross-divergence with a mean-free lattice Poisson
   correction. This correction preserves net circulation and is zero for
   matching FVM/VPM states.
5. A zero blend width retains the conservative hard-lattice transfer and its
   complete release support. The cube candidate uses a three-cell blend band.
6. Advance the VPM with its configured viscous scheme. The transfer does not
   call GBD and supports CS, RWM, DVH, GBD, and NONE through `ViscousConfig`.
7. Resolve the boundary-only panel potential against the current particle
   state before every FVM boundary trace. This is a fixed-time harmonic-state
   refresh; it does not modify particle circulation, advance panel history,
   shed wake, or evolve geometry.

Deleted production algorithms include velocity-defect correction,
conservative-vorticity-defect correction, hybrid-defect correction,
interface-flux injection, scatter/lattice defect transfer, tail-budget logic,
and their configuration modes. They are retained below only as experiment
history so they are not repeated.

## Retained baseline values

| Quantity | Retained value | Evidence |
|---|---:|---|
| FVM domain | `(-3, 3)^3` | Standalone box-convergence error was 0.78%/1.09% in Cd at t=0.05/0.10; the original `(-1.5,1.5)^3` box gave 11.94%/13.42% |
| Refined and transfer box | `(-1.25,1.25)^3` | Original refined region retained; no unsupported xmax/refinement drift |
| FVM outer cell size | `0.25` m | Box-convergence change only; near-body and transfer resolution unchanged |
| FVM time step | `0.005` s | Matches the fresh validation reference |
| VPM/coupling step | `0.010` s | Two exact FVM substeps |
| Boundary mode | `vorticity_mixed` | Dirichlet replacement control changed Cd error 6.145% to 6.150%; no benefit, reverted |
| `eta_blend_width` | `3h = 0.09375` m candidate; `0.0` m historical baseline | Three-cell band has unit/manufactured acceptance; same-seed full-solver acceptance remains pending |
| VPM spacing/core ratio | `h=0.03125`, ratio `1.0`, owned by `ViscousConfig` | The coupler derives both runtime values from the selected VPM viscous configuration; no duplicate coupler knobs |
| GBD threshold | `0.02*h^3`, absolute | Existing physical VPM diffusion; not retuned during replacement validation |
| Pressure anchoring | no coupler implementation or option | The dead no-op pressure-reference code and option were deleted |
| Boundary history after replacement | mandatory internal update | The next FVM interval must start from the replaced current state; the hallucinated optional resynchronization flag was deleted |

## Root causes established by evidence

| ID | Root cause | Evidence | Fix |
|---|---|---|---|
| R1 | Additive defect algorithms retained/added circulation in the overlap instead of imposing one state | Earlier runs showed monotonic VPM enstrophy and `sum(abs(Gamma))`, growing correction clouds, and eventual blow-up; direct conservative variants were contractive but produced 11–59% Cd errors | Delete all defect modes; absolute delete/reinject state replacement |
| R2 | The velocity-defect loop was blind to grid-scale vorticity | Measured transfer gain reaches zero at Nyquist; the old VPM field carried 66.8% of enstrophy above half-Nyquist and reached `max(abs(omega))=1280` versus about 90 in FVM | Removed rather than patched |
| R3 | The original FVM box was too small | Uncoupled `+/-1.5` box differed from the full reference Cd by +11.94%/+13.42% at t=0.05/0.10 | Retain only the evidence-backed `+/-3` coarse outer box |
| R4 | VPM samplers ran before same-time FVM replacement | Before the fix, VPM centreline max error at t=0.05 was 10.10%; sampling the replaced state reduced it to 4.87% with no transfer change | Coupled VPM advance defers scheduled output; coupler samples after replacement and resync |
| R5 | Boundary-only panel strengths were stale relative to the evaluated particle state | Code solved the panel before VPM evolution, then evaluated boundary velocity after evolution/GBD; after FVM replacement it combined new particles with the same old panel strengths. Fixed-time refresh reduced Cd error at t=0.05 from 6.145% to 1.153%, with viscous force and transfer budgets unchanged | Refresh the panel solve from current particles before each boundary trace |
| R6 | The checked-in reference was stale relative to current FVM time step/code | Fresh `dt=0.005` reference Cd is 2.907174/2.172972 at t=0.05/0.10; old checked-in reference was 3.38734/2.27656 | All acceptance numbers below use the fresh reference |

## Controlled experiments and disposition

Drag errors use the fresh `dt_FVM=0.005` fully meshed reference unless noted.

| Experiment | One controlled change | Result | Disposition |
|---|---|---|---|
| Original velocity-defect baseline | None | Good early results with visible time lag; additive high-frequency error accumulated and blew up near t=1 | Rejected/deleted |
| Pedrizzetti magnitude removal, factor 0.30 | Add grid-scale sink | Stable to t=1.65, worst Cd error 37.5% | Reverted/deleted |
| Pedrizzetti magnitude removal, factor 0.15 | Weaker sink | Worst Cd error 18.8% | Reverted/deleted |
| Pedrizzetti rotation only | Preserve strength magnitude | Cd error reached 68.2% at t=0.5 | Reverted/deleted |
| Direct conservative M4-prime defect, `+/-1.5` | Replace velocity correction by `V*omega` defect | Stable/contractive through t=0.2, Cd errors about +59%/+58% at 0.05/0.10 | Rejected/deleted |
| Velocity defect, current code, `+/-1.5` | Control | Cd +6.24%/+16.74%; correction grew to 1.616 by t=0.10 | Rejected/deleted |
| Dirichlet boundary, old defect algorithm | Boundary mode only | Cd 3.0901 versus 3.0885 at t=0.05 | Reverted |
| Downstream bound `xmax=3.5` only | Extend outlet | Cd 3.072 at t=0.05, worse than control | Reverted |
| FVM step 0.010 | Time step only | Cd error -6.55% at t=0.10; correction still grew | Reverted to 0.005 |
| Hybrid defect, `+/-1.5` | Low-band velocity plus high-band conservative defect | Cd +18%/+28.8% at 0.05/0.10 | Rejected/deleted |
| Uncoupled FVM box `+/-2.5` | Domain convergence | Cd +1.97%/+2.34% | Improved but not retained |
| Conservative defect, box `+/-2.5` | Transfer control | Correction contracted 0.568 to 0.432; Cd +11.07%/+11.25% | Rejected/deleted |
| Hybrid defect, box `+/-2.5` | Restore resolved response | Cd +2.87%/+5.37% fresh | Outside gate; rejected/deleted |
| Uncoupled FVM box `+/-3` | Domain convergence | Cd +0.78%/+1.09% | Retained geometry |
| Hybrid defect, box `+/-3`, 1% tail | Old best candidate | Cd +1.39%/+2.98%; faster but still additive algorithm | Deleted at user request |
| Absolute replacement, stale sampler/panel | Requested method, first control | Stable to t=0.10; Cd 6.145%/6.406%; VPM centreline max 10.10%/6.41% | Exposed R4 and R5 |
| Post-replacement sampling | Output order only | At t=0.05, VPM centreline max 10.10% to 4.871%; FVM max 4.882%; Cd unchanged at 6.145% | Retained |
| Dirichlet boundary with absolute replacement | Boundary operator only | Cd 2.728369, 6.150% error versus 6.145% baseline; profiles unchanged | Reverted to `vorticity_mixed` |
| Skip initial t=0 replacement | Startup order only | Cd error 6.145% to 6.132%; profiles unchanged | Insufficient; initial replacement restored |
| Current-particle panel refresh | Harmonic state timing only | Cd error 6.145% to 1.153% at t=0.05; pressure force moved 0.8261 to 0.8985 versus 0.9145 reference; profiles improved slightly | Retained |
| Repeat corrected run to t=0.10 | Stability/reproducibility | t=0.05 Cd repeated within `1.2e-7`; all ten steps stable; metrics below | Accepted |

## Accepted accuracy results

Trial: `/private/tmp/openonda_cube_replacement_panel_refresh_t010_trial`

Reference: `/private/tmp/openonda_reference_dt0005_t1_20260824`

| Time | Metric | Error |
|---:|---|---:|
| 0.05 | Cd | 1.153% |
| 0.05 | FVM centreline max / mean | 4.882% / 0.852% |
| 0.05 | VPM centreline max / mean | 4.805% / 0.276% |
| 0.05 | FVM off-axis max / mean | 1.520% / 0.349% |
| 0.05 | VPM off-axis max / mean | 0.910% / 0.178% |
| 0.10 | Cd | 0.844% |
| 0.10 | FVM centreline max / mean | 3.505% / 0.715% |
| 0.10 | VPM centreline max / mean | 2.310% / 0.225% |
| 0.10 | FVM off-axis max / mean | 1.656% / 0.307% |
| 0.10 | VPM off-axis max / mean | 1.066% / 0.160% |

All measured Cd and velocity-profile metrics at both validated sample times are
below the 5% acceptance limit.

## 2026-08-24 cleanup and calibration preflight

This audit changed no `vorticity_mixed` numerical formula and added no
remeshing/regeneration call.

| Item inspected | Finding / controlled change | Disposition |
|---|---|---|
| Coupler particle spacing and core-radius-ratio options | Duplicated values already owned by the VPM viscous/GBD setup and could drift apart | Deleted from `CouplerSetup`; resolved once from the initialized VPM setup |
| Optional post-transfer resynchronization flag | Allowed a required boundary-history update to be disabled and had no defensible algorithmic meaning | Flag deleted; required update renamed to describe its actual purpose |
| Coupler pressure reference | No-op dead implementation; closed-body force is pressure-datum invariant | File, state, reporting field, and option deleted |
| Boundary-only panel stepping | A pre-evolution panel solve was immediately superseded by the evidence-backed fixed-time solve before the boundary trace | Redundant solve skipped only for `vpm_boundary_condition`; the retained fixed-time panel solve remains active |
| Native VPM checkpoints in coupled tutorials | Duplicated the atomic FVM+VPM checkpoint and could produce ambiguous restart artifacts | Disabled in coupled cube/cylinder/NACA setups; atomic coupler checkpoint retained |
| Trial output selection | Previous helper defaulted to the production case and could append/mix samples | `--case-directory` is mandatory and a fresh trial refuses existing `solution/` or `samples/` |
| Restarted line samples | CSV readers previously treated appended/restarted segments as one history | Readers now retain only the latest monotonically increasing step segment |
| Plot provenance | Stale/mixed checked-in results could be plotted without proving algorithm or time-step identity | Metadata schema/method and accepted FVM timestep are now required |
| Acceptance gate | The integrity script allowed 15% drag and 35% profile errors | Replaced by a strict 5% gate for every coincident Cd and FVM/VPM line-profile maximum |
| Calibration cost metric | Fixed injection-lattice size cannot measure GBD pruning | Matrix uses the next step's pre-replacement VPM cloud plus median step/VPM time |
| Concurrent timing | Reference contention would bias the baseline relative to restart variants | All timing trials run sequentially |
| First B0 matrix execution | B0 completed stably through `t=2.5 s` and copied the exact `t=2` atomic seed, but the harness re-tested the live final checkpoint and falsely reported the seed missing | `_capture_seed` now validates the preserved seed before the advancing live manifest; `--resume` permits only a completed B0/reference workspace |

Preflight verification: 18 focused tests passed; Ruff passed; Pyrefly reported
zero errors for the changed coupler, VPM stepper, validation, and calibration
code. The long-run matrix and its disposition are defined in
`COUPLER_CALIBRATION_PLAN.md`; results will be appended here rather than
changing the retained production baseline in place.

## Verification commands

```text
pytest -q tests/coupler tests/vpm/test_global_regeneration_threshold.py
ruff check source/coupler tests/coupler tutorials/coupled_fvm_vpm
pyrefly check --python-version 3.11 --search-path /opt/anaconda3/envs/OpenONDA/lib/python3.11/site-packages source/coupler
python tutorials/coupled_fvm_vpm/cube_flow/assets/measure_trial_errors.py <trial> <reference>
```

Taichi emitted cache-lock warnings because concurrent processes could not write
its user cache. Tests and simulations completed successfully; the warnings did
not affect solver state or metrics.

## 2026-08-25 panel-solver P0 fixes: full-scope refresh and STL orientation

Two production bugs identified by `PANEL_SOLVER_PROJECT.md`'s P0 requirements
were fixed, plus four defects found in review of the first attempt. Nothing
here changes `vorticity_mixed`, adds remeshing, or touches the FVM-to-VPM
absolute-injection algorithm. Nothing changes the numerical behavior of the
`vpm_boundary_condition` panel mode, which remains the only mode
`cube_flow_setup.py` uses in production.

**Problem 1 — stale `coupling_scope="full"` panel refresh.**
`VPMSolver.refresh_boundary_element_solution()` only re-solved the panel for
`coupling_scope="vpm_boundary_condition"`; for `"full"` it was a silent no-op,
so a coupler that replaces the particle cloud left `full`-scope panel
strengths solved against the pre-replacement state until the next macro VPM
step. This is why the earlier `P1_full_panel` continuation was not a clean
test of synchronized full panel coupling.

Tracing confirmed this was the only lifecycle gap: `physics.body_velocity` /
`_vel()` in `source/solvers/vpm/physics/engine.py` already add panel-induced
velocity to every active particle at every Runge-Kutta stage under `"full"`,
and `compute_induced_velocity_direct` / `_set_coupled_wake_velocity` already
operate on `particles.n_particles_total` with no injected/retained
distinction anywhere in the panel solver.

*Fix.* The guard became `not in ("full", "vpm_boundary_condition")`. Both
coupler refresh call sites in `source/coupler/boundary.py` already called the
method unconditionally, so no new call site was needed. The unit test that
asserted the old skip-for-`"full"` behavior was rewritten.

*Disposition.* Kept. This corrects a scope that is not the current production
configuration for any tutorial, so it carries no risk to the validated
`vpm_boundary_condition` baseline and needs no empirical re-run to accept. It
does **not** decide whether `full` should replace `vpm_boundary_condition` in
production; the same-seed `t=2 -> t=2.5` controlled comparison for that
decision has **not** been run and remains open.

**Problem 2 — geometric-centroid normal-orientation heuristic.**
`add_body_from_mesh_stl` flipped each panel normal based on the direction
from the body's mean panel-centre to that panel — a per-panel test against a
single point, provably wrong for concave bodies. A synthetic concave "long L"
prism (`tests/vpm/test_panel_stl_audit.py`) is watertight, consistently
wound, and correctly outward-oriented, yet the old heuristic flips 4 of its
20 panels.

*Fix.* New
`source/solvers/vpm/boundary_elements/panels/geometry/stl_audit.py`:
`orient_components_by_signed_volume` orients each closed connected component
by the sign of its divergence-theorem volume, correct regardless of
concavity. The same module adds `audit_stl_mesh`, which rejects non-finite
coordinates, degenerate/duplicate triangles, open or non-manifold edges,
inconsistent winding, panel-count overflow, and undeclared multi-body STLs.
`scripts/panel_mesh_audit.py` exposes it as a CLI.

*Disposition.* Kept. Regression-checked against the production
`tutorials/coupled_fvm_vpm/cube_flow/assets/cube.stl`: the new method
produces bit-identical normals to the old heuristic for that STL, and the
same STL passes the audit cleanly (108 triangles, 1 watertight component,
signed volume +1.0). Cube-flow geometry handling is unaffected.

### Defects found in review of the first attempt, and their fixes

| Defect | Evidence | Fix |
|---|---|---|
| Diagnostics evaluated every panel at every particle on every refresh, twice per coupled step, unconditionally — about `108 * 543,276 = 5.9e7` panel-target interactions per refresh for the cube | Measured against the production panel config | Diagnostics are now opt-in (`diagnostic_interval_steps`, default `0` = off), scheduled, and read a deterministic fixed-stride subsample bounded by `diagnostic_sample_size`. Production path measured at **zero** extra evaluations |
| The boundary-face panel diagnostic repeated work the boundary trace already performs, on every trace | Code inspection | Gated behind the same schedule; off by default |
| STL audit ran *after* `_ensure_initialized()`, so an invalid STL still allocated the lattice and the dense influence matrix. The original tests missed this by calling the lower-level loader with a fake lattice | Code inspection | `mesh.py` split into `load_and_audit_body_stl` (no GPU state) and `upload_body_to_lattice`; `add_surface` audits first. Tests now assert `solver.lattice is None` and `solver.aerodynamic_influence_coefficient is None` after rejection |
| `PanelScipySolver.solve` returned `True` unconditionally, so the advertised "hard failure on non-convergence" never fired for the default solver | Reproduced: an inconsistent singular system returned success with relative residual `0.577` and strengths near `1e14` | Both solvers now decide success from the relative residual of the returned solution, with non-finite solutions mapped to infinite residual. Tolerance `1e-8` relative (`DEFAULT_PANEL_RESIDUAL_TOLERANCE`) |
| The reported residual predated the NEUMANN gauge projection, so it did not describe the strengths left in the lattice; BiCGSTAB reported its recursively-updated residual rather than one recomputed from the final solution | Code inspection | `PanelSolver.solve` recomputes the relative residual from the final field state after the projection; BiCGSTAB recomputes from the final `x`. Verified the gauge projection does not inflate it: sphere post-projection relative residual `1.0e-15` |
| `expected_components=2` accepted two shells but uploaded them as a single `PanelBody` with one uid and one kinematics, and oriented every component to positive volume — wrong for nested cavity shells | Code inspection | `PanelSolver.add_surface` now accepts exactly one connected component and rejects multi-component STLs. Genuine per-component bodies and a cavity policy remain unimplemented |
| The `coupling_scope` docstring claimed all four scopes solve in `advance()` once per step and that `vpm_boundary_condition` computes surface pressure — both false | Code inspection | Docstring corrected: `advance()` is skipped entirely for `vpm_boundary_condition`, whose strengths come solely from `refresh_coupled_solution`, which calls `solve()` only and computes no pressure or force |

**Not implemented, and why.** A `no_penetration_residual` diagnostic was
written and then removed: a sphere check produced values around 0.8-5.2
instead of near zero for both NEUMANN and DIRICHLET, despite a linear-solver
residual near `1e-14`. This may be an impulsive-start transient
(`potential_time_derivative` from a zero doublet-strength history on the
first solve) rather than a steady-state defect, but resolving which internal
quantity is the correct total surface velocity for each formulation was not
achievable in this pass. Shipping an unverified diagnostic was judged worse
than shipping none. Consequently the sphere test verifies only the far-field
decay *exponent* and a tight solve residual — not the decay magnitude or
sign, not the analytic surface velocity, and not true error-vs-refinement
convergence.

**Verification.**

```text
pytest tests
ruff check source tests scripts
ruff format --check source tests scripts
pyrefly check source/coupler
python scripts/panel_mesh_audit.py tutorials/coupled_fvm_vpm/cube_flow/assets/cube.stl
```

114 tests pass; Ruff and format clean; Pyrefly reports 0 errors on
`source/coupler`.

**Deferred (not started, not half-implemented).** Coarse-vs-detailed dual-STL
support with hash/provenance metadata; opt-in decimation CLI;
influence-matrix/factorization reuse for static geometry (needs a correctness
audit of the moving-body AIC-rebuild path first — `initialize()` currently
builds the AIC once regardless of kinematics, which may itself be a latent
bug for translating/rotating bodies); a preconditioned iterative solver with
per-iteration convergence history; scalable panel-to-particle evaluation
replacing the direct `O(N_panels * N_particles)` kernel; an accelerated
inside/collision query replacing `absorb_particles` /
`point_inside_stl_body`; an explicit float32-vs-float64 test matrix;
near-surface analytic verification and the ellipsoid/multi-body matrix;
genuine multi-body and cavity support; per-stage panel timing; and the
empirical same-seed cube-flow `t=2 -> t=2.5` decision test.

**Note on coupling priority.** This panel work is orthogonal to the
downstream wake escape/reset mechanism identified as the likely long-time
coupling failure. It does not repair that. The next coupling experiment
should still be the one-step downstream circulation/escape budget, not a
strength correction or broad calibration.

## 2026-08-25 common-lattice blend implementation gate

The production transfer now uses the common M4' lattice for both same-time
states and applies the C1 partition `eta*Gamma_FVM + (1-eta)*Gamma_VPM`.
Mutation is preflighted for capacity and duplicate target nodes. Complete M4'
support conserves signed circulation and componentwise first moments to
floating-point roundoff before the explicit solid-node exclusion.
The blend-only cross-divergence is removed by a lattice Poisson correction;
a manufactured solenoidal test reduces that term to roundoff without changing
net circulation, and a matching-state test confirms the correction is zero.

The focused test gate covers float32/float64 conservation, nonuniform donor
volumes, constant-state reproduction, phase shifts, all six faces, zero and
solid donors, exact fixed points, no double counting at coincident release
nodes, atomic capacity failure, and a manufactured packet that crosses the
ownership face and survives two later replacements. An actual CS step retains
the blended particle count, positions, and vortex strength while satisfying
`sigma^2(t+dt) = sigma^2(t) + 4*nu*dt`.

The VPM scheme API now carries particle spacing and core-radius ratio for CS,
RWM, and NONE as well as DVH/GBD; coupled integration accepts all five schemes.
The cube setup exposes one `VPM_VISCOUS_SCHEME` selector and remains on GBD for
the pending controlled full-solver comparison. Verification at this gate:
`42` coupler tests and `128` repository tests passed; Ruff is clean on the
changed files; focused Pyrefly reports zero errors.
