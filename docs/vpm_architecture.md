# VPM architecture

The VPM runtime advances one inviscid particle state,
`(position, vortex_strength)`, with one coupled explicit Runge--Kutta engine.
Diffusion remains operator-split around that inviscid update.

## Construction

The public construction boundary is the immutable `VPMCase`:

```text
Numerics + initial-condition builders + Backup + Samplers + RunPlan
                         ↓
                       VPMCase
                         ↓
                     VPMSolver
```

`Numerics` is the sole numerical configuration carrier. It selects the RK
tableau, induction method, particle kernel, physical models, device, precision,
capacity, and diagnostic controls. Runtime clocks, particle arrays, and output
destinations are not duplicated in a second numerical setup object.

## Induction and stage information path

```text
VPMSolver.advance()
  └─ EvolutionStepper.advance(dt)
      ├─ optional viscous half-step
      ├─ RungeKutta.advance(..., StageRHS)
      │   └─ each RK stage:
      │       StageState(x_stage, Gamma_stage, core_radius, count, time)
      │         └─ StageRHS.evaluate(...)
      │             ├─ selected InductionMethod.evaluate_stage(...)
      │             └─ external stage contributions
      ├─ optional viscous half-step
      └─ accept step and publish source revision
```

`DirectInduction`, `TreecodeInduction`, and `FMMInduction` implement the same
stage contract. Each receives the complete temporary state and writes velocity,
vortex-strength rate, and an optional auxiliary velocity gradient into
preallocated fields. The RK engine knows none of the backend names or
coefficients beyond its selected tableau.

The documented strength equation is the conservative pairwise transposed
operator with symmetric target/source core regularization. The auxiliary
velocity gradient is a separate diagnostic output; it is not substituted for
the strength rate when unequal core radii make the two discrete operators
different. See [the operator specification](vpm_induction_operator.md).

## Ownership

| Quantity | Canonical owner | Writers | Freshness / invalidation |
| --- | --- | --- | --- |
| Numerical choices | `VPMCase.numerics` | case construction | immutable |
| Position and vortex strength | `Particles` | RK evolution, diffusion, coupling, stabilization, restart | source revision after mutation |
| Core radius and particle geometry | `Particles` | initialization, diffusion, remeshing, coupling, restart | source revision after mutation |
| Velocity, gradient, strain, vorticity | `Particles` | explicit field evaluators | valid only after requested evaluation |
| RK stage fields | `RungeKutta` | one active tableau stage | reused next step |
| Stage rate composition | `StageRHS` | selected induction and external providers | evaluated from supplied stage state |
| Hierarchical workspace | selected induction backend | tree/FMM evaluator | rebuilt or refit for the supplied source state |
| Accepted clock | `VPMSolver` | successful step commit and restart load | staged until the physical step succeeds |
| Output and backup cadence | `OutputManager` | schedule owner | observers consume accepted state |

No observer advances physics or repairs source fields. A failed physical phase
does not publish its staged clock; the solver is terminal-invalid and must be
restarted from the last accepted backup.

## One accepted step

1. Apply pending regeneration and pre-evolution stabilization.
2. Update optional VLM/panel coupling.
3. Evaluate the coupled RK stages through `StageRHS`.
4. Apply viscous diffusion at the configured split points.
5. Apply post-evolution and retention stabilization.
6. Publish the particle source revision.
7. Commit the clock and dispatch accepted-state diagnostics, samples, and
   backups.

The same RK tableau is used for both position and vortex strength. RK2, SSPRK3,
and RK4 are available through the generic `RungeKutta` engine. There is no
fractional integration switch and no separate advection or stretching
integrator.

## Public API

Typical construction is:

```python
from openonda import vpm

case = vpm.VPMCase(
    numerics=vpm.Numerics(
        time_step_size=0.01,
        integrator=vpm.SSPRK3(),
        induction=vpm.DirectInduction(),
        particle_kernel="GAUSSIAN",
        viscous=vpm.ViscousConfig.inviscid(),
    ),
    run=vpm.RunPlan(steps=10),
)
solver = vpm.VPMSolver(case)
solver.run()
```

The old advection, stretching, velocity, and private VPM setup configuration
objects are not public compatibility paths.

## Qualification status

Permanent regression coverage is listed in
[VPM induction-method qualification](vpm_induction_qualification.md). Direct
and treecode stage paths are qualified against independent references; the FMM
hierarchy, kernel contract, and tolerance trend are covered by focused tests.
The current FMM velocity far field uses first-order singular Biot--Savart
multipoles, while its canonical strength rate and auxiliary gradient remain
exact pairwise evaluations. FMM is therefore retained as an explicit opt-in
method rather than silently promoted to the default.
