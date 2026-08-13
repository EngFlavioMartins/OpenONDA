# VPM architecture

This document is the placement guide for the VPM solver. If you are about to
write VPM code — an operator, a diffusion scheme, a stabilization mechanism, a
diagnostic, a sampler, an initial condition or a coupling method — read the
section below that names its subsystem, and put it there. When in doubt, the
directory tree is authoritative.

## Top-level packages

| Package | Responsibility |
|---|---|
| `core/` | Construction and high-level solver lifecycle. `Solver` is a thin facade that wires subsystems and exposes the public API; `EvolutionStepper` owns the per-step time-evolution algorithm. |
| `physics/` | Primary VPM physical operators: induced velocity, stretching, vorticity/strain evaluation, pressure-field evaluation, and the viscous-diffusion schemes (`physics/diffusion/`). |
| `numerics/` | Reusable numerical mathematics with **no VPM policy**: interpolation/remeshing kernels, moment evaluation, constrained projection, and the Taichi kernel factories shared by the physics operators. |
| `stabilization/` | Every algorithm whose purpose is to stabilize, regularize, relax, filter, repair or adapt the particle representation. `StabilizationManager` schedules the workers; each worker implements its own algorithm. |
| `particles/` | Particle storage and primitive particle-set mutation (insert, remove, replace, field transfer). No solver policy. |
| `coupling/` | Orchestration between VPM and the VLM/panel boundary-element solvers during one VPM step. |
| `diagnostics/` | Measurement and reporting only. Diagnostic code must **not** modify the physical state. |
| `io/` | Serialization, backups, logging, export and sampling (`io/sampling/` hosts the field samplers). |
| `config/` | Configuration types and validation only. No runtime state, Taichi fields, filesystem handling or numerical evolution. |
| `initial_conditions/` | Analytic particle fields and canonical vortex initializers (Lamb–Oseen, vortex ring, doublet, Taylor–Green, isotropic turbulence). |
| `kernels/` | Compact interaction-kernel definitions (Gaussian, high-order Gaussian, super-Gaussian, Winckelmans). |
| `acceleration/` | Acceleration structures: the CPU Barnes–Hut treecode, the Taichi LBVH treecode, and the Taichi neighbour search. |
| `turbulence/` | Sub-grid-scale models (Smagorinsky), not stabilization. |
| `boundary_elements/` | The panel/VLM **implementations**. Code that tells VPM how to *use* them lives in `coupling/`. |

## Dependency direction

The allowed edges are (subsystem → subsystem):

```text
core              → physics, stabilization, coupling, diagnostics, io, config,
                    particles, turbulence, boundary_elements, acceleration
physics           → particles, numerics, kernels, acceleration, config, io (logging only)
stabilization     → particles, numerics, diagnostics (read-only), config, io (logging only)
coupling          → physics (public interface), particles, boundary_elements, core(config)
diagnostics       → particles (read-only), physics (read-only evaluation), numerics,
                    config, io (backup reader + logging for offline diagnostics)
io                → config, diagnostics(read-only), particles
config            → (nothing inside VPM, except boundary_elements config types)
initial_conditions → particles (fields only), numpy
particles         → initial_conditions (pure-geometry centerline helper only)
boundary_elements → config, io (loading-distribution sampling output)
coupling          → nothing else
```

`io/logging.py` is a **leaf** logging facility (it imports nothing from VPM); any
subsystem may import it even where a general `io` dependency is not listed.

Forbidden edges:

```text
physics      → stabilization        (stabilization operators are injected/composed, never inherited)
physics      → core
particles    → core
particles    → stabilization
diagnostics  → core / solver policy / state-mutating code
core         → kernels of another subsystem's private state
stabilization → core (workers act on the particle container / public solver interface only)
```

If a change introduces an edge that is not listed, add it consciously and
update this table. Architecture tests fail when a forbidden edge appears.

## Where to put new code

- **Physical operator** (velocity, vorticity, stretching, pressure, strain):
  `physics/`. If it is a self-contained Taichi kernel factory with no VPM
  policy, the kernel factory goes in `numerics/kernels_common.py` and the
  operator that calls it goes in `physics/`.
- **Diffusion method**: `physics/diffusion/` — one module per scheme
  (`core_spreading.py`, `random_walk.py`, and `schemes.py`, which composes
  them). DVH/GBD remain in `physics/diffusion/grid.py` together with the shared
  grid machinery, because both schemes drive one stateful grid; the
  `grid_based_diffusion` / `gbd_diffusion` entry points live there too.
  Register the scheme in the `physics/diffusion/__init__.py` dispatch only if
  dispatch is new; the existing dispatch lives in `core/evolution.py`.
- **Stabilization method**: `stabilization/`, implemented as a worker module
  (`filament_refinement.py`, `divergence_relaxation.py`, `regularization.py`,
  `operators.py`). Register the worker method in the appropriate phase tuple of
  `StabilizationManager.PHASES` (see below); the step loop dispatches phases via
  `run_phase()` and never grows a manual `apply_*()` sequence. Do **not** give
  `physics` a dependency on it.

### Stabilization lifecycle phases

Stabilization acts at fixed points of a time step, and where it may act is
numerically meaningful. `StabilizationManager` owns the schedule in one
table, `PHASES` (a `phase name → ordered worker-method names` mapping), and the
step loop calls `run_phase(name)`. The phases are:

- `pre_evolution` — step entry, before the particle field changes (`capture_reference_state`).
- `pre_strength` — after velocity/gradients and LES residual viscosity are at
  their `t_n` values, before the strength update the relaxation must inform
  (`apply_relaxation`, Pedrizzetti).  A worker that needs the current velocity
  gradient belongs here, where that gradient describes the state being modified.
- `post_evolution` — after advection/stretching/diffusion modified the field,
  while the updated gradients still describe it (`apply_filament_refinement`,
  `apply_divergence_relaxation`, `apply_regularization`).
- `post_step` — end of the step, after diagnostics/IO (`apply_retention`).

Add a new stabilization mechanism by (1) putting its algorithm in a worker
module under `stabilization/`, and (2) registering its entry point in the right
`PHASES` tuple. Keep worker-level policies (frequency, triggers, its own
admissibility rules) inside the worker; the manager only judges the event
against the master criteria.
- **Generic numerical primitive** (a moment sum, a constrained projection, an
  interpolant) used by several subsystems: `numerics/`. The *decision* to use
  it — when and why — stays in the calling subsystem.
- **Diagnostic**: `diagnostics/`. Read-only: it computes and reports, it does
  not mutate particle fields or solver policy.
- **Sampler**: `io/sampling/` (a `Sampler` subclass or a sampling function).
- **Initial condition / analytic flow field**: `initial_conditions/`. Particles
  are returned as numpy arrays; construction of the solver is not its job.
- **Coupling method** (VPM↔VLM/panel exchange): `coupling/`, keeping the actual
  panel/VLM solver implementation in `boundary_elements/`.
- **Configuration knob**: `config/`, in the module of the subsystem it
  configures, and re-exported from `config/__init__.py`.

## physics vs numerics vs stabilization

Three rules that keep these apart:

1. `physics` implements *governing equations*. A physical operator is a term of
   the vorticity-transport equation or a field it needs.
2. `numerics` implements *mathematics with no VPM policy*: it does not know
   about particle stabilization, time-step policy, or solver scheduling. If a
   function decides *whether* or *when* to apply math to stabilize a field, it
   is stabilization (or evolution), not numerics.
3. `stabilization` trades a declared amount of physics for numerical
   robustness. Its algorithms are the *policy*; the primitive moments,
   projections and kernels they need come from `numerics` and `particles`.

## The Solver is a facade

`Solver` composes subsystems and exposes the public API (`VPMSetup`,
`particles`, `physics`, `stabilization`, `io`, field queries, state save/load).
It does not implement new physics. The per-step evolution algorithm — velocity
and gradient preparation, advection/stretching integration, coupled vs
uncoupled integration, operator splitting, viscous dispatch — belongs in
`core/evolution.py` (`EvolutionStepper`). If you are tempted to add a physics
method to `Solver`, put it in `physics/` instead and call it from the stepper.