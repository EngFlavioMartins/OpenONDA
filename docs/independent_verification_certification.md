# Independent Verification and Completion Audit

## 1. Exact source state and environment

The audited implementation is local `development` commit
`d725bef788a95fca71f2d39d126427c74cd14fa7` (`Complete solver architecture
and tutorial remediation`, 2026-09-01T01:08:02+02:00). The test, static,
installed-workflow, and package evidence below was produced from the identical
source tree immediately before that commit; `git diff` confirms that only this
report set and the JUnit evidence file were added afterward.

| Item | Audited value |
| --- | --- |
| OS/kernel | Linux 7.0.0-30-generic, x86_64, Ubuntu 26.04 environment |
| CPU | Intel Core i7-12700H, 20 logical CPUs |
| Python | 3.11.15 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| Taichi / LLVM | 1.7.4 / 15.0.4; CPU `x64` exercised |
| PyVista / VTK | 0.48.4 / 9.6.2 |
| h5py / HDF5 | 3.16.0 / 1.14.6 |
| Ruff / Pyrefly / Pytest | 0.16.2 / 1.2.0 / 9.1.1 |
| GPU | unavailable: `nvidia-smi` is not installed |

## 2. Overall result

The verified local implementation is substantially remediated: the stable CPU
suite passes 416/416 tests, public API and Pyrefly gates report zero violations,
RWM restart is exact, output ownership is consolidated, the native installed
FVM--VPM workflow passes, and package artifacts build and validate.

Formal result: **FAIL**. All 18 full tutorial lifecycles were not run, and the
required macOS/GPU/precision/backend matrix is not available. The specification
does not permit a conditional pass.

## 3. Requirement matrix

| Requirement | Result | Direct evidence |
| --- | --- | --- |
| Exact committed implementation | PASS | Evidence identifies immutable implementation commit `d725bef788a95fca71f2d39d126427c74cd14fa7`. |
| Canonical solver construction | PASS locally | `openonda.vpm.VPMSolver(VPMCase)` and `openonda.fvm.create_fvm_solver(FVMSetup, ...)`; direct FVM solver construction is absent from the public facade. |
| Typed/documented public API | PASS locally | `python scripts/check_api_completeness.py` exits 0; `pyrefly check` reports 0 errors. |
| Hidden numerical environment controls absent | PASS locally | VPM backend no longer reads `OPENONDA_COMPUTE_DEVICE` or `OPENONDA_CPU_THREADS`; compute device is explicit in `Numerics`. |
| One VPM output/cadence owner | PASS locally | `OutputManager.dispatch()` owns accepted-step sampling and backup due checks; `SolverIO` only writes when instructed; compatibility `SamplerExecutor` was removed. |
| One FVM event schedule type | PASS locally | Immutable `RunSchedule` is used by logging, visualization, sampling, and backup; legacy interval fields and `SamplingSchedule` alias were removed. |
| Accepted-state time semantics | PASS locally | `EvolutionStepper` stages step/time, commits only after physical phases, then health/output observe the accepted state. |
| Failed VPM step semantics | PASS locally | A physical exception leaves the accepted clock unchanged and marks `VPMSolver` terminal-invalid; subsequent evolution is rejected. |
| Observer purity | PASS locally | Accepted-state diagnostic refresh recomputes derived velocity/gradient/LES fields without applying residual-viscosity stabilization. |
| Deterministic and stochastic restart | PASS on Linux CPU | Deterministic schemes and counter-based RWM pass uninterrupted-versus-restart equality tests, including fresh-process RWM equality. |
| Atomic/strict backup | PASS locally | Version/fingerprint validation, truncated-file rejection, interrupted-write preservation, and >50,000-particle storage are covered. |
| Numerical qualification | PASS for collected CPU claims | FVM LSQ order 1.96660; Gaussian velocity/gradient errors below `1.11e-7`; treecode order 3.06202; coupler interpolation orders 2.23279 and 2.05562. |
| Full maintained CPU suite | PASS | `solver_verification.xml`: 416 tests, 0 failures/errors/skips, 479.43 s. |
| Static/build gates | PASS | Ruff lint/format, Pyrefly, API completeness, generated-artifact nomenclature, build, Twine, and shell syntax pass. |
| All named tutorial lifecycles | FAIL (unverified) | Installed API smoke passes and prior named entrypoint defects were fixed, but clean/run/plot/clean was not executed for all 18 cases. |
| Linux/macOS, f32/f64, direct/accelerated, GPU matrix | FAIL (unverified) | Only Linux CPU was available in this audit. |

## 4. Architecture ownership table

| Responsibility | Sole owner | Public configuration | Mutable state | Other writers found |
| --- | --- | --- | --- | --- |
| VPM construction/validation | `VPMCase`, `Numerics`, `VPMSolver` | `openonda.vpm` | construction objects are immutable; solver owns runtime | none public; `_to_runtime_setup()` is private |
| FVM construction/validation | `FVMSetup`, `create_fvm_solver` | `openonda.fvm` | solver fields/time after construction | no public direct solver constructor |
| Particle initialization | typed analytical flows/distributions | `VPMCase.initial_conditions` | `Particles` receives one atomic built particle set | initialization objects do not mutate solver incrementally |
| Analytical flow construction | `source/solvers/vpm/initialization/flows` | typed `openonda.vpm` flow objects | immutable model parameters | none |
| Particle distribution | typed distribution models | `openonda.vpm` distribution objects | generated `VortexParticleSet` | none |
| VPM time advancement | `EvolutionStepper` + `VPMSolver` accepted clock | `Numerics.time_step_size`, `RunPlan` | staged then accepted `step/time` | coupler defers output, not clock ownership |
| FVM time advancement | `FVMSolver` | immutable `TimeConfig` and optional `MaximumCourantTimeStep` | solver runtime `time`, `step`, accepted/previous `dt` | no per-call public dt override |
| Velocity/gradient evaluation | `PhysicsEngine` / FVM field operators | typed velocity/scheme config | derived fields/caches | invalidated by particle state revision |
| Viscous evolution | VPM diffusion scheme selected by `ViscousConfig` | `Numerics.viscous` | particle positions/core/strength as scheme requires | no observer writer |
| Stabilization/validity | `StabilizationManager` for modifiers; accepted health monitor for observation | `StabilizationConfig`, `HealthLimits` | explicit stabilization fields/history | diagnostic output does not apply stabilization |
| VPM sampling and backup cadence | `OutputManager` | `Samplers`, `Backup` | sampler indexes; no physical state | none |
| FVM event cadence | `FVMSolver` using immutable `RunSchedule` | `TimeConfig`, `LoggingConfig`, `BackupConfig`, sampler schedules | accepted event timing only | each event writer is called after one due decision |
| Backup restoration | VPM `_BackupIO`; FVM backup modules | solver `load_*` methods | complete restart state after validation | no unrelated config replacement |
| Logging/profiling | solver loggers and profilers | typed logging config | log/profiling records only | no physical-state mutation |
| Coupled checkpointing | `source/coupler/backup.py` | `CouplerSetup.backup_interval_steps` | manifest plus both solver restart artifacts | no component independently writes a coupled checkpoint |

## 5. Step traces

### VPM accepted step

1. `VPMSolver.advance()` rejects use after any prior physical failure.
2. `EvolutionStepper.begin_step()` snapshots accepted step/time into the
   stabilization context and prepares pending topology changes.
3. It stages `step + 1` and `time + dt`; the public solver clock is unchanged.
4. Boundary/VLM/panel coupling runs, then velocity and required gradients are
   evaluated from the current particle revision.
5. LES state and explicit stabilization modifiers are applied at their defined
   physical phases.
6. Advection/stretching and the selected viscous scheme mutate particle state.
7. Post-evolution stabilization runs; particle revision is advanced once.
8. The staged clock is committed. Accepted-state health is evaluated.
9. `OutputManager` dispatches due samples and at most one scheduled backup.
10. A physical exception before commit preserves the accepted clock and marks
    the solver terminal-invalid; observers never receive that failed state.

### Coupled FVM--VPM step

1. The coupler advances the VPM with output deferred.
2. It evaluates the accepted VPM boundary state and advances configured FVM
   substeps, each committing FVM history/time before FVM diagnostics/output.
3. The FVM vorticity is transferred on the common lattice and replaces the
   configured VPM authority region.
4. Boundary history is resynchronized; then the VPM accepted-state health and
   scheduled samplers run against the post-transfer state.
5. Coupler diagnostics/logging record one complete coupling step.
6. Coupled backup cadence writes one manifest plus matching FVM and VPM
   restart artifacts. A failure prevents normal completion/backup publication.

## 6. Public API inventory

The enforceable inventory is the declared `__all__` surface of `openonda`,
`openonda.coupler`, `openonda.fvm`, `openonda.runtime`, and `openonda.vpm`.
`scripts/check_api_completeness.py` inspects every unique exported class,
constructor, function, and class-defined public method for:

- a usable signature;
- annotations for every public parameter;
- no public `Any` parameter/return;
- explicit return annotations; and
- class/method/function documentation.

Final result: **zero violations**. `tests/test_public_api.py` additionally
enforces exact allowlisted solver facades. FVM construction exposes
`create_fvm_solver`, not `FVMSolver`; VPM runtime services and internal setup
models are not public exports.

## 7. Legacy/duplicate paths removed

- VPM `SamplerExecutor` compatibility facade.
- `SolverIO.should_backup` and duplicate solver-side backup cadence checks.
- public `Numerics.to_runtime_setup` (now private `_to_runtime_setup`).
- duplicate Pydantic runtime/state models and the Pydantic dependency.
- public VPM time-step/freestream mutation entry points.
- hidden compute-device and CPU-thread environment overrides.
- FVM legacy `output_interval_steps`, `output_interval_time`, logging interval
  fields, `TimeConfig.steady/transient`, and `SamplingSchedule` alias.
- VPM private tutorial imports and FVM direct-construction tutorial patterns.
- CI `continue-on-error` masking of Pyrefly.

## 8. Test disposition

Every current test file was reviewed. The following disposition is about suite
organization; all remained enabled for the final 416-test run.

| Files | Disposition | Reason |
| --- | --- | --- |
| `tests/coupler/test_common_m4_viscous_lifecycle.py`, `test_lattice_transfer.py`, `test_physical_coupling.py`, `test_stable_renewal.py` | MERGE | Preserve unique invariants but consolidate repeated M4 partition/moment/idempotence cases. |
| `tests/coupler/test_coupled_backup.py` | REWRITE | Retain current restart invariants and extend to a longer real production-mesh continuation. |
| `tests/coupler/test_cube_flow_setup.py`, `tests/fvm/test_cylinder_reference_tools.py`, `tests/tutorials/test_output_schemas.py` | KEEP | Observable case/output behavior now catches current API and schema regressions. |
| `tests/coupler/test_cube_reference_grid_study.py`, `test_flux_handoff.py`, `test_flux_handoff_vpm_integration.py`, `test_fvm_consistency_band.py`, `test_fvm_vpm_smoke.py`, `test_gbd_projected_renewal.py`, `test_gbd_recovery.py`, `test_interpolation_qualification.py`, `test_reporting.py` | KEEP | Each supplies a distinct conservation, transfer, qualification, smoke, or reporting oracle. |
| `tests/fvm/test_curved_cylinder_mesh.py`, `test_geometry_chunks.py`, `test_logging.py`, `test_manufactured_gradient_qualification.py`, `test_mixed_velocity_boundary.py`, `test_restart_and_diagnostics.py`, `test_time_step_control.py` | KEEP | Distinct mesh, output, manufactured-order, boundary, restart, and adaptive-time contracts. |
| `tests/test_lamb_oseen_rwm_statistics.py`, `tests/test_storage_output.py`, `tests/tutorials/test_vortex_interactions.py` | KEEP | Statistical postprocessing, storage schema, and physical tutorial initialization invariants. |
| `tests/test_public_api.py`, `tests/vpm/test_taichi_kernels.py` | MOVE | Their policy checks belong in dedicated static gates; retain until CI invokes those gates directly. |
| `tests/vpm/test_backup_storage.py`, `test_evolution_transaction.py`, `test_output_manager.py` | KEEP | They now cover exact RWM/fresh-process restart, terminal failure semantics, and sole cadence ownership. |
| `tests/vpm/test_panel_diagnostic_scheduling.py` | REWRITE | Replace remaining internal call-count emphasis with emitted-diagnostic and trajectory observables. |
| `tests/vpm/test_sampling_schedule.py`, `test_stabilization_schedules.py` | KEEP | Boundary-value schedule checks complement observable integration coverage. |
| `tests/vpm/test_boundary_element_state_refresh.py`, `test_case_lifecycle.py`, `test_core_numerical_qualification.py`, `test_core_spreading_projection.py`, `test_flow_integral_backend.py`, `test_gbd_prune_conservation.py`, `test_global_regeneration_threshold.py`, `test_health_limits.py`, `test_import_side_effects.py`, `test_logging_cadence.py`, `test_output_configuration.py`, `test_particle_cache_revision.py`, `test_particle_initialization.py`, `test_sampler_execution.py`, `test_state_strict_schema.py`, `test_turbulence_orchestration.py`, `test_vlm_mesh.py` | KEEP | Each tests a distinct public, numerical, cache, initialization, health, output, or state-schema contract. |
| `tests/vpm/test_panel_far_field.py`, `test_panel_linear_solver_convergence.py`, `test_panel_moving_qualification.py`, `test_panel_multibody.py`, `test_panel_particle_coupling.py`, `test_panel_solver_memory_guard.py`, `test_panel_solver_sphere_analytic.py`, `test_panel_stl.py`, `test_panel_stl_audit.py` | KEEP | Independent analytic, convergence, invariance, memory, coupling, and geometry evidence. |

No test is xfailed or skipped. No known scientific defect is hidden by an
expected failure.

## 9. Numerical qualification results

| Claim | Independent reference and norm | Measured result |
| --- | --- | --- |
| Coupler interpolation | affine exactness and relative L2 refinement | affine max `1.1102230246251565e-16`; L2 `0.0729920`, `0.0155289`, `0.00373539`; orders `2.23279`, `2.05562` |
| FVM spatial gradient | manufactured solution, relative L2, 8/16/32 grids | `0.0996837`, `0.0255046`, `0.00641315`; minimum order `1.96660` |
| Gaussian Biot--Savart | closed-form velocity/gradient, relative L2 | velocity `9.6116e-08`; gradient `1.1090e-07` |
| Barnes--Hut convergence | direct sum, relative L2 at theta .8/.4/.2 | `0.159585`, `0.0230220`, `0.00228808`; order `3.06202` |
| Deterministic VPM restart | uninterrupted versus save/destroy/load/continue | Euler/RK2/RK3/RK4 and NONE/CS/DVH/GBD pass configured equality bounds |
| RWM reproducibility/restart | counter-based seed, accepted step, particle, component | same-process, fresh-process, and split restart are exactly equal on CPU |
| FVM adaptive stepping | manufactured CFL scaling and variable-step BDF polynomial exactness | immediate CFL reduction, 1.2 growth cap, max-dt ceiling, exact event/end-time landing, quadratic-exact BDF all pass |

These measurements are valid for the recorded Linux CPU environment. They do
not substitute for the missing cross-platform/backend matrix.

## 10. Restart and reproducibility

Passing CPU evidence includes deterministic VPM integrators/viscous schemes,
counter-based RWM fresh-process and restart equality, FVM restart and diagnostic
history, coupled backup manifest/hash checks, >50,000-particle backup,
truncated-file rejection, and interrupted-write preservation. Backup loading
validates format and configuration before replacing solver state.

Not independently established here: macOS versus Linux, every accelerated/GPU
backend, f32 versus f64 scientific equivalence, and a long production-scale
coupled continuation.

## 11. Tutorial result

All 18 maintained tutorial directories have `setup.py`, `allrun.sh`,
`allplot.sh`, and `allclean.sh`; all shell scripts pass `bash -n`. Private VPM
imports, hidden numerical backend controls, stale coupled-cube configuration,
and the vortex-ring nonexistent/duplicate plotting calls were corrected.
`python scripts/validate_native_tutorials.py --compute-device CPU` passes the
installed public FVM--VPM solve, output, coupled backup, and restore workflow.

Formal tutorial certification remains FAIL because all 18 named cases were not
run through clean/run/plot/clean in disposable directories. In particular, the
cylinder grid study and the four 1,200-step vortex-interaction production runs
are now user-ready but were not completed as part of this audit.

## 12. Commands and results

| Command | Result |
| --- | --- |
| `ruff check source tests tutorials scripts openonda` | PASS |
| `ruff format --check source tests tutorials scripts openonda` | PASS, 474 files formatted |
| `pyrefly check` | PASS, 0 errors (narrow Taichi/PETSc suppressions documented in configuration) |
| `python scripts/check_api_completeness.py` | PASS, zero public API violations |
| `python scripts/check_nomenclature.py --paths --generated` | PASS, canonical nomenclature scan completed against the final source tree |
| `python -m build` | PASS after isolated build dependencies were allowed |
| `twine check dist/*` | PASS for wheel and sdist |
| `python -m pytest tests -p no:cacheprovider --junitxml=solver_verification.xml` | PASS: 416 passed, 0 failed/errors/skipped, 5 warnings, 479.43 s |
| `python scripts/validate_native_tutorials.py --compute-device CPU` | PASS |
| `find tutorials ... bash -n` | PASS for every shell entrypoint |

The five Pytest warnings are one upstream Taichi deprecation and four JUnit
xunit2 `record_property` warnings; no test was skipped or xfailed.

## 13. Files changed during verification

The remediation touched the public facades, FVM/VPM/coupler configuration and
runtime paths, tests, CI/static scripts, documentation, environment metadata,
and maintained tutorial setups/scripts. Principal additions are:

- `scripts/check_api_completeness.py`;
- `source/solvers/fvm/config/scheduling.py`;
- `source/solvers/fvm/core/time_step.py`;
- `tests/fvm/test_time_step_control.py`;
- the three certification reports; and
- `solver_verification.xml`.

The implementation and removal of stale generated result baselines were
committed as `d725bef788a95fca71f2d39d126427c74cd14fa7`.

## 14. Final certification statement

CERTIFICATION: FAIL. The implementation is not yet complete.

Blocking certification items, in priority order:

1. Execute all 18 named tutorial clean/run/plot/clean lifecycles, including the
   four-resolution cylinder study and all advertised VPM variants.
2. Complete the required Linux/macOS, f32/f64, direct/accelerated, and maintained
   GPU backend equivalence matrix.
