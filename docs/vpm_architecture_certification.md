# VPM Architecture Verification and Completion Certification

## 1. Status and scope

**NOT CERTIFIED**

The local VPM architecture now satisfies the audited ownership, cadence,
failure, cache, observer, and Linux-CPU restart checks. Formal certification is
withheld because the required cross-platform/GPU/precision matrix and complete
tutorial lifecycle evidence do not exist.

- audited implementation: `d725bef788a95fca71f2d39d126427c74cd14fa7`
- environment: Linux 7.0.0-30, x86_64, Python 3.11.15, NumPy 2.4.6,
  SciPy 1.17.1, Taichi 1.7.4, h5py 3.16.0/HDF5 1.14.6
- exercised backend: Taichi CPU; no available CUDA/Vulkan/Metal/macOS host

## 2. Public architecture map

```text
openonda.vpm construction objects
  VPMCase
    Numerics + HealthLimits + StabilizationConfig
    typed analytical flow/distribution initial conditions
    Backup + Samplers + RunPlan
        ↓ private construction translation and validation
VPMSolver
  accepted clock and lifecycle
  Particles (canonical particle fields/count/revision)
  PhysicsEngine (velocity/gradient/viscous evaluation)
  EvolutionStepper (physical phase order and staged clock)
  StabilizationManager (explicit modifiers)
  OutputManager (sole cadence/path/index owner)
        ↓ accepted state only
diagnostics / samples / logging / atomic backup
```

The public workflow is:

```python
case = vpm.VPMCase(...)
solver = vpm.VPMSolver(case)
solver.run()
```

Interactive and external-coupling use may call `advance()`; they do not create
a second configuration path. Internal `VPMSetup` translation is private and is
not an `openonda.vpm` export.

## 3. Configuration ownership

| Quantity | Authoritative owner | Derived/backend representation | Result |
| --- | --- | --- | --- |
| time-step size | immutable `Numerics.time_step_size`; restored runtime value in solver | `EvolutionStepper.time_step_size` read-only working value | PASS |
| accepted time and step | `VPMSolver.time`, `VPMSolver.step` | staged clock inside `EvolutionStepper`; `RestartState` is load input only | PASS |
| flow model | validated `Numerics.flow_model`; copied resolved runtime description | `PhysicsEngine` receives resolved choice | PASS |
| viscous scheme | `ViscousConfig` | selected diffusion implementation | PASS |
| advection/stretching | immutable typed configs | stepper scheme dispatch | PASS |
| particle kernel | `Numerics.particle_kernel` | captured Taichi kernel functions | PASS |
| compute device/precision | `Numerics.compute_device` and precision | initialized Taichi runtime | PASS; no hidden env override |
| domain bounds/capacity | `Numerics` | particle container allocation | PASS |
| molecular viscosity | typed viscous/initial-condition config | per-particle kinematic-viscosity field | PASS; consistency validated before run |

No public time-step or freestream setter remains. The coupler uses a private
freestream synchronization method because it is a physical modifier inside the
coupled algorithm, not user configuration.

## 4. Mutable state-ownership matrix

| State | Canonical owner | Writers | Readers | Invalidation | Restarted? |
| --- | --- | --- | --- | --- | --- |
| time / step | `VPMSolver` | accepted-step commit; validated backup load | stepper, health, output, coupler | schedules observe only after commit | yes |
| particle count | `Particles.n_particles_total` | encapsulated add/remove/regenerate/load methods | kernels, output, capacity checks | device mirror and topology workspaces updated inside mutation methods | reconstructed/validated |
| position | `Particles.position` | advection, RWM, stabilization, coupling, load | physics/tree/output | `touch_state()` after committed mutation | yes |
| velocity | `Particles.velocity` | physics evaluation/load | advection, diagnostics, output | derived refresh on source revision | yes, then refreshable |
| vortex strength | `Particles.vortex_strength` | stretching, stabilization, coupling, load | physics/diagnostics | source revision | yes |
| core radius | `Particles.core_radius` | viscous schemes, refinement/remeshing, load | physics/health | source revision | yes |
| particle volume | `Particles.particle_volume` | initialization, remeshing/coupling, load | diffusion/diagnostics | source revision | yes |
| molecular viscosity | `Particles.kinematic_viscosity` | initialization/load | diffusion/effective viscosity | source revision when changed | yes |
| eddy/effective viscosity | particle fields | LES/viscous evaluation/load | viscous update/diagnostics | recomputed at intended physical phase | yes |
| velocity gradient / strain / vorticity | particle derived fields | `PhysicsEngine`/stepper diagnostic refresh | stretching, LES, diagnostics | recomputed after source revision; gradient recomputed on load | vorticity yes; gradient derived |
| group ID / zone ID | particle fields | initialization, topology/coupling, load | diagnostics, region logic | topology mutation contract | yes |
| RWM random state | no mutable stream | counter function keyed by seed, accepted step, particle index, component | RWM displacement kernel | no stream invalidation required | seed/config + accepted step suffice |
| diffusion counters | solver/stepper | diffusion scheme and load | scheduling/scheme state | explicit on diffusion/restore | yes |
| particle state revision | `Particles._state_revision` | `touch_state()` in mutation boundary | cached particle properties/tree snapshot | monotone source invalidation | reset safely after load, not scientific state |
| topology revision | not separate | topology mutations also advance state revision and rebuild capacity/workspaces | topology-dependent workspaces | same encapsulated mutation boundary | not applicable as restart state |
| stabilization diagnostics | `StabilizationManager` | stabilization/health restore | logging/backup | accepted-step context | selected fields yes |
| solver diagnostic history | `VPMSolver._diagnostics_history` | diagnostic recorder | CSV export | observer-only | full exported history is not demonstrated as restart-critical |

The host/device count pair is a necessary backend mirror. Callers do not
manually synchronize it; all population mutations encapsulate both updates and
cache invalidation.

## 5. Particle mutation and cache contract

All physical mutations route through `Particles` methods or stepper-owned
backend kernels, followed by one published particle-state revision. Population
changes update host/device count, capacity, active slices, and dependent
workspaces inside the owning operation. External code is not required to call a
`modify/sync/touch/invalidate` sequence.

Verified cases:

- vortex-strength mutation at unchanged time invalidates the cached source
  snapshot;
- position mutation at unchanged time cannot reuse the previous tree state;
- population changes rebuild topology-dependent views;
- the cache key is particle state revision, never simulation step alone.

`tests/vpm/test_particle_cache_revision.py` and the complete 416-test run pass.

## 6. Time integration and failure semantics

Execution order in `EvolutionStepper.advance()` is:

1. begin stabilization context and apply pending regeneration;
2. stage the next clock without publishing it;
3. advance VLM/panel boundary physics when configured;
4. evaluate velocity and required velocity gradient at the intended state;
5. update LES fields and apply explicit pre-strength stabilization;
6. perform coupled or split advection/stretching;
7. apply viscous diffusion at the configured split point;
8. apply post-evolution/post-step stabilization;
9. publish the particle revision;
10. commit the staged clock;
11. evaluate accepted-state health and dispatch observers/output.

Supported configurations covered by the suite include fractional/coupled
integration, Euler/RK2/RK3/RK4 paths used by maintained cases, stretching
enabled/disabled, inviscid, CS, RWM, DVH, and GBD. Invalid combinations fail in
construction validation.

If a backend/physical phase raises after partial device mutation, the accepted
clock is not committed and `VPMSolver._evolution_failure` makes the instance
terminal-invalid. A second `advance()` raises and instructs the caller to build
a new solver from the last accepted backup. This is explicit defined-state
failure handling rather than silent continuation.

## 7. Observer/modifier separation

| Subsystem | Classification | May mutate primary physical state? |
| --- | --- | --- |
| advection/stretching/diffusion | modifier | yes, at explicit step phases |
| stabilization/refinement/remeshing | modifier | yes, through narrow phase contracts |
| VLM/panel/coupler updates | modifier | yes, at explicit coupling phases |
| health diagnostics | observer | no |
| scientific samplers | observer | no |
| logging/profiling | observer | no |
| VTK/HDF5 visualization | observer | no |
| numerical backup writer | observer | no |

Accepted-state diagnostic refresh recomputes disposable velocity, gradient,
vorticity, and LES-derived fields but no longer applies residual-viscosity
feedback. Stabilization's `update_residual_viscosity()` is called only in the
physical evolution path. Sampler exceptions propagate with step/time context
instead of changing the solver mode.

## 8. Output and restart architecture

`OutputManager` is the sole VPM runtime owner of:

- accepted-step, initial, and final output events;
- sampler schedule decisions;
- backup due decisions;
- sample directories and time-series indexes.

`VPMSolver._write_backup()` and `SolverIO.write_backup()` perform the requested
write without rechecking cadence. The former `SamplerExecutor` compatibility
facade and `SolverIO.should_backup()` were deleted.

Backup format 10.0 stores the accepted clock, time-step size, numerical
configuration fingerprint, freestream, diffusion/regeneration counters,
particle fields, group/zone IDs, filament lineage where applicable, and
stabilization diagnostic state. Writes use a temporary file plus atomic replace;
strict validation rejects unknown, missing, incompatible, or truncated input
before state replacement.

Counter-based RWM removes opaque generator state. Each Gaussian draw is a pure
function of the declared seed and stable accepted-step/particle/component
identifiers, making uninterrupted and restarted continuation identical on the
same backend.

## 9. Construction/API/documentation verification

- `openonda.vpm` has an explicit allowlisted facade.
- Runtime services (`OutputManager`, `SolverIO`, internal setup models) are not
  public exports.
- `Numerics._to_runtime_setup()` is private.
- public construction parameters and methods are documented and annotated.
- `scripts/check_api_completeness.py` reports zero violations across all
  `openonda` facades.
- `pyrefly check` reports zero errors; only narrow configured Taichi/PETSc
  suppressions remain.
- no Pydantic dependency or duplicate Pydantic state model remains.
- no `OPENONDA_COMPUTE_DEVICE`/`OPENONDA_CPU_THREADS` override remains.

## 10. Numerical and restart evidence

| Verification | Result |
| --- | --- |
| Gaussian Biot--Savart closed form | velocity relative L2 `9.6116e-08`; gradient `1.1090e-07` |
| Treecode versus direct sum | errors `0.159585 → 0.0230220 → 0.00228808`; order `3.06202` |
| Same-step cache invalidation | dedicated source-revision regression passes |
| Deterministic continuation | Euler/RK2/RK3/RK4 and NONE/CS/DVH/GBD split-run comparisons pass |
| RWM same process | exact equality with identical seed/configuration |
| RWM fresh process | exact equality with identical seed/configuration |
| RWM restart | uninterrupted and save/destroy/load/continue states are exactly equal |
| Large backup | >50,000-particle save/load passes |
| Corrupt/interrupted backup | truncated input rejected; prior complete file survives simulated interrupted write |
| Full stable suite | 416 passed, 0 failed/errors/skipped, 479.43 s |

## 11. Quality commands

| Command | Result |
| --- | --- |
| `ruff check source tests tutorials scripts openonda` | PASS |
| `ruff format --check source tests tutorials scripts openonda` | PASS |
| `pyrefly check` | PASS, 0 errors |
| `python scripts/check_api_completeness.py` | PASS |
| `python -m pytest tests -p no:cacheprovider --junitxml=solver_verification.xml` | PASS, 416/416 |
| `python scripts/validate_native_tutorials.py --compute-device CPU` | PASS, including coupled backup/restore |
| `python -m build`; `twine check dist/*` | PASS |

No test is skipped or xfailed. The JUnit artifact is
`solver_verification.xml`.

## 12. Remaining limitations

### Blocking formal certification

1. CUDA/Vulkan/Metal and macOS results were not available.
2. Cross-precision and direct/accelerated scientific equivalence is incomplete.
3. Every VPM tutorial and variant was not run through clean/run/plot/clean.
4. A production-scale long coupled restart and the complete exported diagnostic
   history across restart remain outside the demonstrated matrix.

### Non-blocking implementation notes

- Particle topology currently shares the monotone source-state revision. A
  distinct topology revision is unnecessary for correctness today because all
  topology-dependent workspaces are rebuilt by the mutation boundary; it would
  become necessary if independently cached topology products are introduced.
- Treecode is intentionally approximate; the measured convergence is acceptable
  on the tested CPU backend, not a universal backend certification.

## 13. Changes made

The verification/remediation pass changed VPM construction, runtime, diffusion,
backup, output, API, typing, tests, tutorials, CI/static gates, and these reports.
It added counter-based RWM and exact restart tests, removed duplicate output and
configuration paths, and enforced terminal failure semantics. The implementation
is committed as `d725bef788a95fca71f2d39d126427c74cd14fa7`.

## 14. Final statement

The VPM architecture is committed, coherent, and passes the available
Linux-CPU evidence, but it is not formally certified because mandatory
platform/backend/tutorial evidence remains incomplete.

CERTIFICATION: FAIL. The implementation is not yet complete.
