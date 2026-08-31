# VPM architecture

This document records the VPM information-flow contract as it exists in the
solver.  It is deliberately about ownership and time levels, rather than the
implementation details of individual Taichi kernels.

## Construction

The public construction boundary is the immutable `VPMCase`:

```text
Numerics + initial-condition builders + Backup + Samplers + RunPlan
                         ↓
                       VPMCase
                         ↓
                     VPMSolver
```

`Numerics` defines physics, numerical methods, device and precision.  It does
not contain a clock, particle arrays, or an output destination.  `VPMSetup` is
the private adapter used by the established numerical engine; new user code
should construct a `VPMCase`, not modify a solver after construction.

## Ownership

| Quantity | Canonical owner | Writers | Principal readers | Freshness / invalidation |
| --- | --- | --- | --- | --- |
| Immutable numerical choices | `VPMCase.numerics` | case construction only | solver, physics, stabilization | immutable |
| Restart and log destinations | `VPMCase.backup` | case construction only | backup and logging runtime | immutable |
| Scientific sampler declarations and schedules | `VPMCase.samplers` | case construction only | output manager | immutable |
| Particle primary state: position, vortex strength, core radius, volume, molecular viscosity, IDs | `Particles` | initialization, advection, stretching, diffusion, coupling, stabilization | physics, coupling, diagnostics, samplers | `Particles.touch_state()` advances `state_revision` after source changes |
| Particle derived fields: velocity, velocity gradient, strain rate, vorticity, eddy/effective viscosity | `Particles` | field evaluation and turbulence/stabilization operators | physics, diagnostics, samplers | valid only after the explicitly requested evaluator has run for the required state |
| Spatial acceleration hierarchy | `PhysicsEngine` / treecode | physics evaluation | target velocity/gradient evaluation | keyed to `Particles.state_revision` and particle count |
| Scalar clock and step | `VPMSolver` | accepted evolution commit; restart load | coupling, schedules, I/O, diagnostics | committed only after all physical phases succeed |
| Stabilization lineage and event diagnostics | `StabilizationManager` | stabilization workers; backup load | stabilization, backup | reset/rebuilt with particle topology; serialized in a restart |
| Diagnostic histories | `VPMSolver` | diagnostics recorder | export/reporting | observers only; never source state |

`Particles` owns the only mutable particle-resolved representation.  Public
container operations (`add`, `replace`, masked source updates and removal)
maintain the active count and call `touch_state()`.  Raw device kernels are an
internal exception: the evolution transaction publishes one source revision
after all such physical updates have completed.

## One accepted time step

The algorithm is orchestrated by `EvolutionStepper.advance()` in this order:

1. Apply any pending regeneration and pre-evolution stabilization.
2. Advance VLM/panel coupling when enabled.
3. Evaluate velocity and, when required, velocity gradient at the stage state.
4. Update LES/residual-viscosity fields.
5. Advance advection and stretching (jointly for coupled RK); apply viscous
   splitting/diffusion.
6. Run post-update, post-evolution, and retention stabilization phases.
7. Publish a new particle source revision and atomically commit the clock,
   then run observers/output at the completed particle state.

For fractional integration, stretching uses the gradient prepared before the
strength update.  Coupled RK evaluates stage fields inside the physics engine;
its public contract is that positions and vortex strengths are advanced at the
same RK stages.  Core spreading is split symmetrically around this coupled
update.

## Freshness and observer rules

Velocity and gradient are derived fields, not alternate particle state.  A
caller that requires current fields must ask the orchestration layer for an
explicit field evaluation before it observes them.  Sampling, logging,
diagnostics and export may read particle fields and diagnostic histories, but
must not advance physics, repair particle data, or change the source revision.

The treecode cache is source-state based, not step based: two source mutations
within one step must result in distinct `state_revision` values before a target
query may reuse an acceleration structure.  CPU snapshots must not be treated
as mutable particle state.

## Stabilization and coupling

`StabilizationManager` owns phase scheduling and its lineage data.  Its workers
receive a typed `StabilizationStepState`, the latest immutable metric values,
and a `ParticleMutationPort`; they never receive solver getters/setters or an
unbounded solver reference.  They may change particle fields only through that
canonical mutation port.
`CouplingStepper` is likewise a modifier only when it injects/removes/deflects
particles.  VLM and panel diagnostics are observers after their coupled update.

## Backup and restart

A backup is a numerical checkpoint, not visualization output.  It stores the
clock, particle-resolved fields and IDs, freestream state, diffusion and
stabilization history, plus a fingerprint of the numerical configuration.
Loading validates the file and configuration before it mutates a solver.  A
restart is required to reproduce an uninterrupted continuation to the expected
floating-point tolerance; write precision used for visualization is not allowed
to reduce restart precision.

## Migration guardrails

During the transition from `VPMSetup` to `VPMCase`, do not add new mirrors of
numerical configuration or particle fields on `VPMSolver`.  New dependencies
should be explicit (`self.solver.particles`, `self.solver.physics`, and so on),
not acquired through broad forwarding.  Both evolution and coupling steppers
therefore list their consumed capabilities explicitly.  CPU particle snapshots
are revision-keyed, and physics modules communicate optional events through a
dependency-neutral observer interface; logging presentation is injected from
the orchestration/I/O side.

An unsuccessful physical phase never publishes the staged time or step.  Its
partially executed device work is intentionally not presented as an accepted
state; recovery is the caller's responsibility, just as it is for a failed
kernel launch.
