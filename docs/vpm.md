# Vortex-particle solver guide

This page is the reader-facing contract for OpenONDA's vortex-particle method (VPM)
and its optional vortex-lattice/panel components. The public construction objects are
available from [`openonda.vpm`](../source/solvers/vpm/__init__.py). The implementation
uses Taichi fields for the active particle prefix and keeps the accepted physical clock
in `VPMSolver`.

## Start here

Build a case from immutable numerical controls, one or more declarative initial
conditions, output policy, and a finite run plan:

```python
from openonda import vpm

distribution = vpm.RectangularDistribution(
    bounds=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
    spacing=0.25,
    core_radius_ratio=2.5,
)
case = vpm.VPMCase(
    directory="first-vpm-case",
    numerics=vpm.Numerics(
        time_step_size=1.0e-2,
        precision="f32",
        induction=vpm.DirectInduction(stretching_scheme="TRANSPOSED"),
    ),
    initial_conditions=(
        vpm.VortexFilament(
            vortex_core_radius=0.35,
            circulation=1.0,
            kinematic_viscosity=1.5e-5,
            distribution=distribution,
        ),
    ),
    run=vpm.RunPlan(steps=2),
)

solver = vpm.VPMSolver(case)
solver.run()  # run() writes terminal output and closes owned resources
```

For interactive or coupled control, construct the solver, call `advance()` or the
explicit sampling methods, and call `close()` in a `finally` block. `run()` is a
single-owner lifecycle and may be called only once. Initial-condition builders are
evaluated exactly once, immediately before the first requested evolution or run event.

## Terminology and units

OpenONDA uses the following terms consistently:

* A **vortex particle** is one quadrature point carrying a position, regularization
  radius, volume, velocity, and vector **vortex strength**.
* **Vorticity** is `ω` and has units 1/s. For the particle representation,
  `ω_i = Γ_i / V_i`.
* **Vortex strength** or **circulation vector** is `Γ_i = ω_i V_i` and has units
  m³/s. It is not a scalar circulation unless a model explicitly projects it onto a
  filament direction.
* **Core radius** (also called `sigma` in the kernel equations) is the physical
  regularization length in m. `core_radius_ratio` means `sigma / h`, where `h` is the
  nominal particle spacing.
* A **particle volume** is the quadrature weight `V_i` in m³; it is not necessarily
  the geometric volume of a rendered point.

| Quantity | Shape | Unit | Owner/meaning |
| --- | --- | --- | --- |
| `position` | `(N, 3)` | m | Active particle coordinates. |
| `velocity` | `(N, 3)` | m/s | Particle transport velocity, including freestream/external contributions after refresh. |
| `vortex_strength` (`Γ`) | `(N, 3)` | m³/s | Vector strength used by Biot--Savart induction. |
| `vorticity` (`ω`) | `(N, 3)` | 1/s | Derived/diagnostic field `Γ / V`. |
| `core_radius` (`sigma`) | `(N,)` | m | Per-particle kernel radius. |
| `particle_volume` (`V`) | `(N,)` | m³ | Per-particle integration weight. |
| `kinematic_viscosity` (`nu`) | `(N,)` | m²/s | Molecular viscosity assigned to each particle. |
| `eddy_viscosity` / `effective_viscosity` | `(N,)` | m²/s | LES contribution and `nu + nu_t`. |
| `velocity_gradient` (`J`) | `(N, 3, 3)` | 1/s | `J_ij = ∂u_i/∂x_j`. |
| `strain_rate` | `(N, 3, 3)` | 1/s | Symmetric part `(J + Jᵀ)/2`. |
| `time` / `time_step_size` | scalar | s | Accepted physical clock and macro-step. |

`Particles` allocates a fixed `max_n_particles` capacity at construction. The active
population is the prefix `[0:N)`, where `N == particles.n_particles_total`; unused
capacity is not part of a solution. Device fields (`particles.position`, etc.) are
Taichi storage. The corresponding `*_cpu()` accessors return active-prefix NumPy
copies/cached views with the shapes above. Mutating source fields, changing positions,
strengths, radii, or population requires `touch_state()` or a solver mutator so cached
host data and acceleration structures are invalidated.

The immutable `ParticleDistribution` and `VortexParticleSet` used by initializers copy
their input arrays, validate exact shapes and finite values, and mark the arrays
read-only. A solver insertion copies those arrays into its device fields; later solver
evolution is mutable.

## Induction and vortex stretching

For a source particle at `x_j` and a target at `x_i`, the regularized Biot--Savart
operator computes velocity from `Γ_j`, displacement `r = x_i - x_j`, and the selected
radial kernel. Particle-to-particle interactions use the symmetric radius
`(sigma_i + sigma_j)/2`; arbitrary target evaluation uses the source radius. The
velocity gradient is `J_kl = ∂u_k/∂x_l`.

`Numerics.induction` selects the computational path. `stretching_scheme` independently
selects the strength-rate formulation:

| Backend | What it does | Precision/support trade-off |
| --- | --- | --- |
| `DirectInduction` | Exact all-pairs regularized velocity/gradient and strength-rate evaluation, O(N²). | Supports f32/f64, all four public radial kernels, and CPU/Vulkan/CUDA/Metal Taichi devices. |
| `TreecodeInduction` | LBVH/Barnes--Hut traversal with hierarchical velocity/gradient approximation. | Device-resident; currently f32 and Gaussian/Winckelmans kernels only. |
| `FMMInduction` | Fixed-order device FMM (`P2M → M2M → M2L → L2L → L2P`) plus exact kernel-specific near-field P2P. | Device-resident CPU/Vulkan path; currently f32 only. |

For the current Jacobian convention the choices are:

```text
DIRECT      : dΓ/dt = J   @ Γ
TRANSPOSED  : dΓ/dt = J.T @ Γ
MIXED       : dΓ/dt = 0.5 * (J + J.T) @ Γ
```

The default is `TRANSPOSED`. The formulation is part of the case identity and is
recorded in backend diagnostics and solver metadata. It is not implied by the backend name.
Direct induction evaluates the selected contraction during its pair walk. Treecode
and FMM contract the hierarchical gradient; their rates therefore inherit the
approximation error of the gradient. `strength_rate_enabled=False` still evaluates
velocity (and any requested gradient) but writes zero strength rate, which is useful
for diagnostic refreshes.

`FMMInduction.evaluate_targets` currently uses the shared regularized target kernels
for arbitrary target locations because the production FMM workspace has a particle
target pass but no dual-tree arbitrary-target pass. This is an explicit documented
fallback; it does not change the FMM path used for particle RK stages.

## Kernels and core radius

`GAUSSIAN`, `WINCKELMANS`, `HIGH_ORDER_GAUSSIAN`, and `SUPER_GAUSSIAN` provide the
dimensionless radial functions `q(r/sigma)` and `zeta(r/sigma)`. `q` includes the
`1/(4*pi)` Biot--Savart factor; `zeta` is the normalized radial vorticity profile.
`RadialVortexKernel` also exposes the pair velocity, pair gradient, conservative
transposed pair rate, far-field errors, and tolerance-based cutoffs used by tree/FMM
near-field decisions. Kernel choice affects the regularization and, consequently, the
resolved core, diffusion compatibility, and backend support.

The ratio `sigma/h` is a numerical resolution choice, not a unit conversion. A smaller
ratio sharpens the represented vortex but increases sensitivity to particle spacing;
the requested physical core may be corrected by `ParticleCoreCompensation` in supported
analytical initializers. Do not compare `vortex_core_radius` and `core_radius` without
checking whether the initializer is asking for a physical or represented core.

## Time integration and state ownership

`RungeKutta` advances position and vortex strength with the same explicit tableau
(`RK2`, `SSPRK3`, or `RK4`). At each stage it builds temporary stage position and
strength fields from the accepted state and previously computed stage rates; the
accepted particle fields are not replaced until the final weighted combination:

```text
accepted (xⁿ, Γⁿ)
    └─ stage 0..s-1: temporary (x_stage, Γ_stage) → RHS (u, dΓ/dt)
    └─ final RK combination: mutate persistent (xⁿ⁺¹, Γⁿ⁺¹)
```

Core spreading, random-walk diffusion, and grid-based DVH/GBD diffusion are operator-
split by `EvolutionStepper` after/beside the coupled inviscid update. Gaussian core
spreading uses symmetric half-steps around the RK update. DVH may accumulate accepted
steps until its resolved heat-kernel interval is reached. GBD/DVH regeneration can
replace the entire particle cloud, including positions, strengths, radii, volumes,
viscosities, and IDs; the replacement is an accepted-state mutation and invalidates
all source caches.

External stage providers (freestream, VLM, panel, or coupling callbacks) receive the
temporary stage state and may add velocity, a gradient, or an explicit strength rate.
They must not read an older accepted state through the stage protocol. The VLM/panel
boundary solve may be lagged to the accepted coupling phase while its velocity is
evaluated at the exact temporary particle positions.

## Initial conditions and particle mutation

`RectangularDistribution`, `CylindricalDistribution`, `ToroidalDistribution`, and
`TriangularPrismDistribution` create immutable geometry/quadrature. Flow builders such
as `VortexFilament`, `VortexRing`, `VortexDoublet`, `TaylorGreenVortex`, and
`IsotropicTurbulence` attribute velocity, strength, viscosity, and optional group/zone
IDs. Call `build()` directly to inspect a `VortexParticleSet`, or place builders in
`VPMCase.initial_conditions` and let the solver insert them.

`VPMSolver.add_vortex_particles` appends validated arrays; the batch fields must have
the same length `N` and shapes `(N, 3)`/`(N,)`. `replace_vortex_particles` replaces the
active prefix and is the operation used by grid diffusion/remeshing. Removal methods
compact every aligned field, change the active count, and invalidate source state.
`update_particle_vortex_strength` and `set_particles_properties` are supported runtime
mutations; use them between accepted stages, not while an RK stage is evaluating.

## Viscosity, turbulence, and stabilization

`ViscousConfig` selects no diffusion (`NONE`), core spreading (`CS`), random walk
(`RWM`), diffusion via a vortex heat-kernel grid (`DVH`), or grid-based diffusion
(`GBD`). `RWM` assumes spatially uniform effective viscosity. `DVH` also has a scalar
viscosity/heat-grid contract; `GBD` is the path for LES variable effective viscosity.
The solver validates the relevant `nu`, spacing, kernel, grid, and time-step criteria
at case construction/runtime boundaries.

`TurbulenceConfig` selects DNS or LES closure. LES updates particle `eddy_viscosity`
from the strain-rate field and stores `effective_viscosity = nu + nu_t`. Stabilization
limits and diagnostics are evaluated around accepted steps; configured health failures
either raise or stop the finite run according to `RunPlan.health_limit_action`.
They are not substitutes for grid or time-step convergence studies.

## Sampling, backups, and restart

`EverySteps`, `EveryTime`, and `FinalOnly` are immutable schedules. `EverySteps` fires
on an accepted step cadence. `EveryTime` fires once when an accepted state crosses a
physical-time boundary; it does not interpolate between states. `FinalOnly` is dispatched
only by the framework's final event. Initial output is opt-in on each sampler through
its `initial` setting where supported.

`SurfaceSampler` evaluates a planar structured grid and `LineSampler` evaluates a
regular line. `sample()` returns canonical scalar columns (`position_x`, …,
`velocity_*`, `vorticity_*`, and optionally the six independent strain components plus
the nine gradient components). Coordinates are m, velocity m/s, and derivatives 1/s.
`save_csv()` writes one snapshot; framework-owned scheduling appends restart-safe CSV
time/step rows. `save_vtp()` for these structured samplers writes VTK StructuredGrid
(`.vts`) despite the historical method name.

`FlowIntegralsSampler` and `RingDiagnosticsSampler` provide higher-level diagnostics;
their source classes define the additional columns and applicability requirements.
Missing prerequisite data is either rejected or logged as a skipped sampler according
to the sampler contract; a sampler write failure is fatal to the owning lifecycle.

Global integral diagnostics are reported per unit density: kinetic energy is in
m⁵/s², its time rates are in m⁵/s³, helicity is in m⁴/s², and enstrophy is in
m³/s². Net vector circulation, linear impulse, and angular impulse are in m³/s,
m⁴/s, and m⁵/s respectively. These are unbounded-domain VPM integrals, not joules
or newton-seconds unless the documented density/reference factors are applied.
The VPM ``total_enstrophy`` convention is ``integral(|omega|²) dV`` without the
one-half factor used by the FVM diagnostic.

`Backup` controls numerical restart cadence and directories. The current VPM default is
the historical `solution/` directory under `VPMCase.directory`; `Samplers` always
writes below `samples/` (optionally below a validated relative subdirectory). Backups
are not scientific samples. The solver writes `vpm_metadata.json` in that same backup
directory at construction, run startup, checkpoint writes and lifecycle completion.
During a run it reports `running` with the latest recorded checkpoint state;
manual checkpoints report `partial` until the solver closes. Closing a manually
advanced solver also records its accepted state. Infinite
configuration bounds use the JSON strings `Infinity` and `-Infinity` in metadata
and restart fingerprints; the numerical inputs retain their floating-point values.
Load a compatible backup into a newly constructed `VPMSolver`; a failed physical
evolution is deliberately terminal for that solver instance. A continuation may
increase `max_n_particles` when filament refinement and regularization are disabled.
This reserves more storage while retaining the saved state and physical settings.
Smaller allocations and capacity changes with either adaptive operator enabled
still require an exact capacity match.

## VLM and panel coupling

`Numerics.vlm` attaches a vortex-lattice solver whose bound-vortex field is solved in
the accepted coupling phase and added to particle RK stages. `PanelSolver`/`PanelBodySetup`
attach surface panels and body kinematics. Both paths have geometry/provenance and
viscosity consistency checks. Use [`docs/coupling.md`](coupling.md) for the coupled
FVM/VPM driver, transfer regions, and boundary-trace semantics.

For a coupled particle wake, `VLMSetup(wake_core_overlap=2.5)` sets the initial
core radius to 2.5 times the larger of the local span spacing and wake-row
length. The circulation solve and particle emitter use the same radius for
both trailing and transverse elements. Omitting this option preserves the
legacy radii; `sigma_factor` scales only the transverse elements in that
legacy rule. Choose overlap together with spatial/time resolution and verify
the resulting forces and wake, rather than treating it as a stability switch.

`ForceConfig.kutta_joukowski(unsteady=True)` adds the unsteady Bernoulli
pressure term to the bound-leg loads. The solver differences the cumulative
surface potential jump over the accepted physical step and integrates force
and moment over each panel, split at its bound line. This is a backward time
difference; resolve startup and prescribed acceleration with the time step.
The default `unsteady=False` retains quasi-steady loads. The option changes
reported loads, not particle transport or the circulation solve.

Native force and chordwise CSV samples include the separate `unsteady_force_*`
components, while `force_*`/`panel_force_*` contain the configured total.
The sampled pressure jump includes the pressure-time contribution. VTK also
stores `unsteady_panel_force` and the intrinsic `panel_moment_correction` about
the bound midpoint; integrated moments and per-surface torque/power include it.
`VLMSetup.logging_interval_steps` controls force/distribution CSV cadence
independently of `VLMSampler` geometry output and `Backup` cadence.
The pressure term does not provide separation, stall or viscous skin friction.

## Practical limits

The direct path is the best correctness/reference choice for small clouds and for
qualification. Treecode/FMM reduce interaction cost but require backend/precision/
kernel compatibility and numerical tolerance studies. Particle capacity is fixed;
choose `max_n_particles` for the full run, including shedding and diffusion regeneration.
The current public evidence does not make every GPU backend or every stabilization/
diffusion combination a universally qualified production choice. Inspect diagnostics,
health limits, and `vpm_metadata.json` for the selected case rather than inferring support
from an accepted configuration alone.
