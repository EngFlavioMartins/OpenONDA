# FVM/VPM convention alignment: work plan

Target conventions: `docs/development/conventions.md`. Restart compatibility
with old backups is not kept (user decision). Remove replaced names; do not
keep aliases or compatibility branches (zero-legacy policy). Decisions taken
by the user: Smagorinsky coefficient default 0.17 in both solvers; VPM
particle frames get their own `Output` schedule, independent of backups;
string options lowercase; samplers record the initial state by default.

Work happens in the worktree `/home/flavio-martins/Projects/OpenONDA-conventions`
(branch `conventions-alignment`). Run Python there with
`PYTHONPATH=/home/flavio-martins/Projects/OpenONDA-conventions`. Never edit the
main checkout `/home/flavio-martins/Projects/OpenONDA`.

## Phase 1: shared core (package `openonda/core`)

C1. `core/schedules.py` (done): `EverySteps`, `EveryTime`, `FinalOnly`,
    `Schedule`, `schedule_from_dict`, `OutputSchedule` protocol. Change rule:
    `EveryTime` is never due at step 0 (already true for `EverySteps`).
C2. `core/status.py`: `RunStatus(StrEnum)`: `not_started`, `created`,
    `running`, `completed`, `not_converged`, `partial`, `failed`, `unstable`,
    `resolution_lost`, `wall_time_limit`.
C3. `core/output_events.py`: `OutputEvent(StrEnum)`: `initial`,
    `accepted_step`, `final`, `failed`.
C4. `core/file_output.py`: `write_text_atomic`, `write_json_atomic`,
    `append_csv_rows(path, columns, rows)` (append; validate header when the
    file exists), `rewind_csv(path, time)`. `core/vtk_output.py` gains
    `read_collection`, `write_collection` (shortest round-trip `repr(float)`
    times, `group="" part="0"` omitted), `rewind_collection`. Replace the five
    PVD writers and both atomic writers; `core/restart.py` must stop importing
    `openonda.fvm.io.solver_io`.
C5. `core/sampling.py`: `FIELD_SAMPLE_COLUMNS` (time, x, y, z, velocity and
    vorticity components), `Sampler` base class with `config_dict()`,
    keyword-only `file_name`, `schedule=EverySteps(1)`, `initial=True`,
    `line_points(start, end, n_points=None, spacing=None)` (ceil rule,
    exactly one argument) and `surface_points(point, normal, bounds, spacing)`
    (bounded-uniform grid: both bounds included).
C6. `core/profiling.py`: `Profiler(*, enabled, synchronize=None, parallel=None,
    output_path=None)` with `section(name)` context manager,
    `begin_step(step, time, time_step_size)`, `end_step()`, `report()`;
    optional MPI reduction and resident-memory sampling (from
    `fvm/io/profiling.py`), optional device synchronization (from
    `vpm/io/runtime_profiler.py`).
C7. `core/log_style.py`: `run_header(solver_name, precision, *rows)` replaces
    both `print_openonda_header`; `configure_logging(name, path, *, console,
    level)` configures a standard-library logger with the shared formatter.
C8. `core/metadata.py`: `write_run_metadata(path, *, solver, case_name,
    status, configuration, sections)`: shared keys, solver adds sections.

## Phase 2a: FVM package (`openonda/fvm`, `tests/fvm`, `tutorials/fvm`)

F1. Public case API is `FVMCase` only. `FVMSetup` becomes the internal
    `FVMConfiguration` (not exported). `FVMSolver(case)`; delete
    `create_fvm_solver` and `fvm/factory.py` (mesh build/load moves into the
    solver or `FVMCase`). `FVMSolver` always sets `self.case`. Remove the
    `Backup`/`Output` aliases: rename `BackupConfig` -> `Backup`,
    `OutputConfig` -> `Output`, `PimpleControl` -> `PimpleConfig`;
    delete `TimeConfig` and `RunSchedule`.
F2. Case fields per conventions: `name, directory, variant, numerics,
    initial_conditions, run, output, samplers, backup, logging`, then `mesh,
    mesh_quality, boundaries, velocity_boundaries, cores, mesh_motion`.
    `Numerics` gains `time_step_size` and `time_step_control`
    (`MaximumCourantTimeStep | None`, formerly `RunPlan.adjustment`);
    `Numerics.coupling` -> `Numerics.pimple`; `Numerics.logging` moves to the
    case. `RunPlan(end_time, start_time=0.0, initial_output=True,
    final_output=True)`; `RunPlan.output_schedule` -> `Output.schedule`.
    `Backup(schedule=None, write_at_end=True)`; no `path` field. `run()` loses
    `initial_velocity` (use `InitialFields`).
F3. Backups: numbered `solution/fvm/fvm_{step:06d}.npz` selected by
    `core/restart.py`, like the VPM. Digest keys `format_version`,
    `configuration_sha256`, `mesh_sha256`. `fvm/io/backup.RestartState` ->
    `FieldRestartState`.
F4. Methods: `save_state`/`load_state` -> `save_backup(path=None) -> Path` /
    `load_backup(path)`; `solve_pimple` -> `solve_step`; `advance_time` ->
    `accept_step(*, defer_output=False, time=None)`; `advance(*,
    defer_output=False, time=None)`; `write_accepted_step_output` ->
    `write_step_output`; `write_vtk` stays; `close(*, failure=None)`;
    `_save_automatic_backup` -> `_save_scheduled_backup`,
    `_automatic_backup_state` -> `_last_backup_state`; public
    `state_revision` property; `n_procs()` -> `n_ranks` property; MPI-collective
    `get_*` accessors in `fvm/coupling/coupler_interface.py` -> `gather_*`,
    cheap local ones -> properties (audit which are collective);
    `set_dirichlet_velocity_boundary_condition_vec` -> drop `_vec`; typed
    `AcceptanceLimitError(RuntimeError)` for rejected candidates; chain
    exceptions in `_collective_io_failure`; `run_status` uses `RunStatus`
    (`completed`, interactive stepping is `running`).
F5. Samplers: `fvm/sampling` adopts `core/sampling.py` and core schedules;
    signature order per conventions; `k` -> `n_neighbours`; surface grid
    bounded-uniform; VTS through `core.vtk_output.write_vtk_dataset` with
    `Output.precision` and `time` field data, no magnitude arrays (check
    tutorial assets first); `save_vts` -> `write_vtk`; delete `strict`, delete
    unused `samples_dir()` helper; dispatcher uses `OutputEvent`.
F6. Logging: standard-library logger `openonda.fvm` via `core.log_style`;
    keep solver report helpers in `fvm/io/logging.py` as noun-named functions
    (`courant_number`, `forces`, ...), `begin_step`/`end_step`,
    `section(title, *rows)`; delete `Timer` in favour of `core.profiling`.
    Profiling via `core/profiling.Profiler` (delete `fvm/io/profiling.py`).
    Metadata via `core/metadata.py`.
F7. Turbulence: `TurbulenceConfig` constructors `dns()`, `smagorinsky()`,
    `equilibrium_smagorinsky()`; one lowercase spelling per model, no aliases;
    default `dns()` (not `None`); C_s default 0.17. Classes end in `Model`
    (`SmagorinskyModel`, `EquilibriumSmagorinskyModel`).
F8. Layout: split `fvm/config/types.py` by concept (as `vpm/config/`);
    diagnostics into `fvm/diagnostics/`; sorted `__all__`; no banners; all
    string options lowercase (`"pimple"`, `"piso"`, `"simple"`, ...); config
    dataclasses frozen + slots; remove defensive `getattr(self, "_x", d)`.
F9. Update `tests/fvm/*`, `tutorials/fvm/*` and the FVM-only
    `tutorials/coupled_fvm_vpm/*/reference_flow/setup.py`.

## Phase 2b: VPM package (`openonda/vpm`, `tests/vpm`, `tutorials/vpm`)

V1. Case fields per conventions: add `variant` and `output`, `logging`;
    `RunPlan(end_time, initial_output=True, final_output=True,
    state_limit_action="raise", wall_time_limit_seconds=None,
    runtime_compute_device=None)` replaces `steps`, `initial_samples`,
    `final_backup`; `Backup(schedule=None, write_at_end=True)` replaces
    `interval_steps`, `directory`, `log_directory`; `Samplers(samples)` loses
    `directory`; `Numerics.write_precision` -> `Output.precision`;
    `Numerics.verbose`, `debug_mode`, `diagnostics` -> `LoggingConfig`.
V2. Particle frames written on `Output.schedule`, independent of backups
    (`solution/vpm/vpm_{step:06d}.vtu` + `vpm.pvd`); backups stay
    `solution/vpm/vpm_{step:06d}.h5`. Digest keys `format_version`,
    `configuration_sha256`.
V3. Solver: `__enter__`/`__exit__`; `close(*, failure=None)`;
    `execute_scheduled_samplers` -> `write_step_output`; `save_backup(path=None)
    -> Path` and `load_backup(path, ...)` public, delete `_save_backup_to`,
    `_load_backup_from`; delete `RestartState` and `_sync_restart_state`;
    `_set_freestream_velocity` -> `set_freestream_velocity`;
    `set_particles_properties` -> `set_particle_properties`; delete
    `export_diagnostics_csv`; `export_state` -> `write_state`; public
    `write_run_metadata(*, status)`; `flush_output()`; `state_revision`
    property; remove `setup` alias (use `numerics`); `backup` attribute ->
    `_backup_config`; public `solution_dir`, all directories `Path`;
    `_init_*`/`_setup_vlm_solver` -> `_initialize_*`;
    `_ensure_evolution_usable`/`_mark_evolution_failure` as in the FVM, and
    `load_backup` clears the failure; `start_from` hint `str | Path`;
    `run_status` uses `RunStatus`; private renames
    (`_initial_n_particles_total` -> `_n_initial_particles`,
    `_is_particle_regeneration_pending` -> `_particle_regeneration_pending`);
    remove defensive `getattr(self, "_x", d)`.
V4. Sampling: move to `vpm/sampling/{fields,flow_integrals,vlm,manager}.py`;
    adopt `core/sampling.py`, `OutputEvent`, core CSV/PVD helpers; signature
    order per conventions (`LineSampler` gains `n_points` and `initial`);
    `save_vtp` (writes `.vts`) and `save_vtk` -> `write_vtk`; `save_csv` ->
    `write_csv` without `# time=` lines; flow-integral CSV written by
    `FlowIntegralsSampler`.
V5. Logging: standard-library logger `openonda.vpm` via `core.log_style`;
    remove the stdout/stderr tee; `begin_step`/`end_step`,
    `section(title, *rows)`, f-strings; profiling via `core/profiling.Profiler`
    (delete `vpm/io/runtime_profiler.py`; fix `_record(time_step_size)`
    naming); metadata via `core/metadata.py`, file `solution/vpm_metadata.json`.
V6. Turbulence: constructors `dns()`, `smagorinsky()`, `sigma()`,
    `equilibrium_smagorinsky()`, `inviscid()`; lowercase model names; C_s
    default 0.17; `turbulence/turbulence.py` -> `turbulence/les.py`,
    `ParticlesLES` -> `ParticleLES`.
V7. Names: `VLMSetup` -> `VLMConfig`, `VLMSurfaceSetup` -> `VLMSurface`,
    `VLMMeshSetup` -> `VLMMeshConfig`; all string options lowercase (compute
    devices, kernels, viscous schemes, stretching schemes, threshold modes,
    state-limit actions); config dataclasses frozen + slots; sorted
    `__all__`; no banners; no environment variable set at import.
V8. Update `tests/vpm/*`, `tests/support/vpm/*`, `tutorials/vpm/*` (setups
    and assets that read metadata or samples).

## Phase 3: coupler, coupled tutorials, shared tests, docs

K1. `FVMVPMCoupler` + `create_coupler` -> `CoupledSolver(fvm_case, vpm_case,
    config)`; `CouplerSetup` -> `CouplerConfig`; `backup_interval_steps` ->
    `backup: Backup`; use only public solver members (no `_resolved_setup`,
    `_time_config`, `_state_revision`, `_save_backup_to`, ...).
K2. Update `tutorials/coupled_fvm_vpm/*`, `tests/coupler/*`, top-level
    `tests/*.py`, `openonda/verify_install.py`, `docs/*.md`, READMEs.

## Phase 4: shared rigid-body motion

M1. After the moving-mesh work is committed: one set of motion classes in
    `openonda/core/kinematics.py` (`StaticMotion`, `TranslatingMotion`,
    `RotatingMotion`, `PitchingMotion`, `HeavingMotion`,
    `LinearPeriodicMotion`, `SmoothRampMotion`, `ManeuverMotion`,
    `CompositeMotion`), frozen dataclasses with exact closed-form increments,
    used by the FVM mesh motion and the VLM surfaces; delete the `*VLM`
    kinematics classes.

## Verification

Every phase ends with ruff, a full import of `openonda.fvm`, `openonda.vpm`
and `openonda.coupler`, and the affected tests. Run GPU tests one at a time.
At the end: the whole test suite, and a short run of every tutorial.
