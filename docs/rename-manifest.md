# OpenONDA rename manifest

Companion to `docs/nomenclature.md`. Every identifier being renamed during the
nomenclature refactor is listed here with its scope so the work can be tracked
and verified mechanically (see PR9's sweep list in the plan).

Scope key:

- **API** — public `openonda` facade and documented constructor/config names.
- **internal** — attributes and helpers reachable from solver internals/tests.
- **serialized** — checkpoint/restart files, CSVs, HDF5 datasets, VTK names.
- **kernel** — inside tight numerical loops; equation symbols permitted.

Deprecation policy: **no aliases**. Each rename is a hard break; all call sites
are updated in the same change.

## 1. Solver classes and factories

| Old | New | Units | Scope |
|---|---|---|---|
| `Solver` (FVM) | `FVMSolver` | – | API |
| `Solver` (VPM) | `VPMSolver` | – | API |
| `FVMVPMCoupler.__init__(vpm_solver, fvm_solver, setup)` | `(fvm_solver, vpm_solver, coupler_setup)` | – | API |
| `setup_fvm_solver` | `create_fvm_solver` | – | API |
| `setup_vpm_solver` | `create_vpm_solver` | – | API |
| `setup_coupler` | `create_coupler` | – | API |
| constructor arg `config:` | `setup:` | – | API |
| `self.config` (FVMSolver, VPMSolver, FVMVPMCoupler) | `self.setup` | – | internal |
| `build_config()` (tutorials) | `create_fvm_setup()` | – | API |
| `make_vpm_setup()` (tutorials) | `create_vpm_setup()` | – | API |

`self.config` intentionally stays on classes that are not the three solver/coupler
owners above: `VorticityTransfer`, `StabilizationContext`/`StabilizationManager`,
and `PostProcess` (offline replay driver, `source/solvers/FVM/sampling/postprocess.py`)
each cache their own subsystem-scoped config reference — `VorticityTransfer`
already has an unrelated `setup()` method, so renaming its attribute would
collide. `SnapshotContext`, the sampler-facing peer `PostProcess` constructs to
mimic a live solver, does use `self.setup` — the sampler duck-type contract
(`source/solvers/FVM/sampling/forces.py`) reads `context.setup`, not `.config`.

Compatibility policy: **no aliases**, confirmed already in effect — neither
`openonda.fvm`/`openonda.vpm` export a bare `Solver`, nor do `setup_fvm_solver`
/`setup_vpm_solver`/`setup_coupler` exist anywhere in the public facade.
`tests/test_public_api_has_no_legacy_aliases.py` guards this.

## 2. Stepping API

| Old | New | Units | Scope |
|---|---|---|---|
| FVM `evolve()` | `advance()` | – | API/internal |
| VPM `update_state()` | `advance()` | – | API/internal |

## 3. Time vocabulary

| Old | New | Units | Scope |
|---|---|---|---|
| FVM `TimeConfig.delta_t` | `time_step_size` | s | API/internal |
| VPM `time_step_size` | `time_step_size` (unchanged) | s | API/internal |
| runtime `flow_time` | `time` | s | internal |
| runtime `time_step` | `step` | – | internal |
| coupler `self.t_end` | `self.end_time` | s | internal |
| coupler `self.dt_fvm` | `self.fvm_time_step_size` | s | internal |
| coupler `self.dt_vpm` | `self.vpm_time_step_size` | s | internal |
| `n_fvm_substeps` | `fvm_substeps` | – | internal |
| `backup_frequency` | `checkpoint_interval_steps` | – | API |
| `coupler_backup_period` | `checkpoint_interval_steps` | – | API |
| `backup_file_name` | `checkpoint_name` | – | API |
| `BackupSystem` | `CheckpointManager`/`CheckpointIO` | – | internal |

## 4. Fluid properties

| Old | New | Units | Scope |
|---|---|---|---|
| FVM `TransportConfig.nu` | `kinematic_viscosity` | m²/s | API |
| VPM `ViscousConfig.viscosity` | `kinematic_viscosity` | m²/s | API |
| coupler `self.nu` | `self.kinematic_viscosity` | m²/s | internal |
| coupler `self.rho` | `self.density` | kg/m³ | internal |

## 5. FVM fields and field state

| Old | New | Units | Scope |
|---|---|---|---|
| `self.U` | `self.velocity` | m/s | internal |
| `self.p` | `self.kinematic_pressure` | m²/s² | internal |
| `self.phi` | `self.face_flux` | m³/s | internal |
| `self.U_old` | `self.velocity_old` | m/s | internal |
| `self.U_old_old` | `self.velocity_older` | m/s | internal |
| `self.phi_old` | `self.face_flux_old` | m³/s | internal |
| `self.phi_old_old` | `self.face_flux_older` | m³/s | internal |
| `self.nut` | `self.eddy_viscosity` | m²/s | internal |
| `initial_p` | `initial_kinematic_pressure` | m²/s² | internal |
| `FieldState` fields | same vocabulary | – | internal |

## 6. Mesh vocabulary

| Old | New | Units | Scope |
|---|---|---|---|
| `n_elements` | `n_cells` | – | internal |
| `element_centroids` | `cell_centroids` | m | internal |
| `element_volumes` | `cell_volumes` | m³ | internal |
| `element_faces` | `cell_faces` | – | internal |
| `element_ids` | `cell_ids` | – | internal |
| `elem_*` prefixes | `cell_*` | – | internal |
| boundary `startFace` | `start_face` | – | serialized/import adapter |
| boundary `nFaces` | `n_faces` | – | serialized/import adapter |
| boundary `neighbourPatch` | `neighbour_patch` | – | serialized/import adapter |

## 7. Boundary conditions

| Old | New | Units | Scope |
|---|---|---|---|
| `type_velocity` | `velocity_type` | – | API |
| `value_velocity` | `velocity_value` | m/s | API |
| `type_p` | `pressure_type` | – | API |
| `value_p` | `kinematic_pressure_value` | m²/s² | API |
| `type_phi` | `flux_type` | – | API |
| `type_nut` | `eddy_viscosity_type` | – | API |

## 8. Configuration classes and controls

| Old | New | Units | Scope |
|---|---|---|---|
| `MeshConfig` | `MeshQualityConfig` | – | API |
| `SchemesConfig` | `DiscretizationConfig` | – | API |
| `ExecutionConfig` | `ComputeConfig` | – | API |
| `OutputSetup` | `OutputConfig` | – | API |
| `LogConfig` | `LoggingConfig` | – | API |
| `DynamicMeshConfig` | `MeshMotionConfig` | – | API |
| `alpha_u` | `velocity_relaxation` | – | API |
| `alpha_p` | `pressure_relaxation` | – | API |
| `momentum_maxiter` | `momentum_max_iterations` | – | API |
| `amg_maxiter` | `amg_max_iterations` | – | API |
| `*_tol` | `*_tolerance` | – | API |
| `Cs` | `c_s` | – | API |
| `Ck` | `c_k` | – | API |
| `Ce` | `c_e` | – | API |
| `Cw` | `c_w` | – | API |

## 9. VPM particles (live, serialized, methods)

| Old | New | Units | Scope |
|---|---|---|---|
| `circulation` (Taichi field) | `vortex_strength` | m³/s | internal |
| `circulation_cpu()` | `vortex_strength_cpu()` | m³/s | internal |
| `radius` | `core_radius` | m | internal/serialized |
| `viscosity` | `kinematic_viscosity` | m²/s | internal/serialized |
| `viscosity_turbulent` | `eddy_viscosity` | m²/s | internal/serialized |
| `viscosity_effective` | `effective_viscosity` | m²/s | internal/serialized |
| `grad_u` | `velocity_gradient` | 1/s | internal |
| `ParticlesState.velocities` | `velocity` | m/s | serialized |
| `ParticlesState.strengths` | `vortex_strength` | m³/s | serialized |
| `ParticlesState.radii` | `core_radius` | m | serialized |
| `ParticlesState.volumes` | `volume` | m³ | serialized |
| `ParticlesState.viscosities` | `kinematic_viscosity` | m²/s | serialized |
| `ParticlesState.viscosities_t` | `eddy_viscosity` | m²/s | serialized |
| `ParticlesState.group_ids` | `group_id` | – | serialized |
| `ParticlesState.vorticities` | `vorticity` | 1/s | serialized |
| local `gamma` / `strengths` | `vortex_strength` | m³/s | kernel |

## 10. VPM setup and state

| Old | New | Units | Scope |
|---|---|---|---|
| `characteristic_distance` | `particle_spacing` | m | API |
| `particles_kernel` | `particle_kernel` | – | API |
| `processing_unit` | `compute_device` | – | API |
| `max_targets` | `max_evaluation_points` | – | API |
| `vpm_domain_bounds` | `domain_bounds` | m | API |
| `regen_*` | `regeneration_*` | – | API |
| `dvh_rd_ratio` | `dvh_support_radius_ratio` | – | API |
| `SetFlowModel` | `set_flow_model` | – | internal |
| `CachedParticleProperty` | `cached_particle_property` | – | internal |
| `self.LES` | `self.turbulence_model` | – | internal |
| `number_of_particles` | `n_particles` | – | internal |
| `num_sources` | `n_sources` | – | internal |
| `_cached_step` | `_cache_step` | – | internal |
| `E_previous` | (removed — dead) | – | internal |
| `E_previous2` | (removed — dead) | – | internal |

## 11. Coupler

| Old | New | Units | Scope |
|---|---|---|---|
| `self.fvm` | `self.fvm_solver` | – | internal |
| `self.vpm` | `self.vpm_solver` | – | internal |
| `self.transfer` | `self.vorticity_transfer` | – | internal |
| `transfer_region_box` | `transfer_region_bounds` | m | API |
| `bc_patch_name` | `coupling_patch` | – | API |
| `vpm_bc_mode` | `boundary_condition_mode` | – | API |
| `overlap_zone_dead_zone_width` | `vpm_only_width` | m | API |
| `overlap_zone_ramp_width` | `authority_ramp_width` | m | API |
| `_u_bc_prev` (etc.) | `_previous_boundary_velocity` (etc.) | m/s | internal |

## 12. Sampling and output

| Old | New | Units | Scope |
|---|---|---|---|
| ad-hoc sampling cadence | `SamplingSchedule` | – | API |
| `logging_interval_steps` | `logging_interval_steps` | – | API |
| CSV column names | singular, unit-honest vocabulary | – | serialized |

Serialized keys deliberately still spelled with the pre-rename vocabulary,
deferred to PR7 so writers and readers move together:

| Key | Where | Paired reader |
|---|---|---|
| `"dt"` | FVM checkpoint, partitioned restart, step log, forces CSV | same modules |
| `"flow_time"`, `"time_step"` | VPM HDF5 `solver` attrs, XDMF, sampler CSV headers | `BackupSystem`, `SamplerExecutor` |

The diagnostics JSONL already writes `time_step_size` (the `StepDiagnostics`
field was renamed in PR2); `PostProcess._archived_timesteps` accepts both
`time_step_size` and the legacy `dt` spelling so pre-rename archives still
load. `SolverState.time` / `.step` already accept the legacy `flow_time` /
`time_step` spellings via `validation_alias`, so pre-rename restart files still
load.

## 13. Tutorials

| Old | New | Scope |
|---|---|---|
| `RHO` | `DENSITY` | tutorial |
| `NU` | `KINEMATIC_VISCOSITY` | tutorial |
| `U_INF` | `FREESTREAM_VELOCITY` (vector) | tutorial |
| `U_INF_MAG` | `FREESTREAM_SPEED` (scalar) | tutorial |
| `L_CUBE` | `CUBE_SIDE` | tutorial |
| `DX` / `H` | `CELL_SIZE` | tutorial |
| `DS` / `H_PARTICLES` | `PARTICLE_SPACING` | tutorial |
| `N_MAX` | `MAX_PARTICLES` | tutorial |
| `T_END` | `END_TIME` | tutorial |
| `LineSampler as FVMLineSampler` alias imports | namespace imports | tutorial |
| `Solver as FVM_Solver` alias imports | `FVMSolver` | tutorial |
| `cubeFlow` | `cube_flow` | path |
| `boundaryLayer` | `boundary_layer` | path |
| `lambOseenVortex` | `lamb_oseen_vortex` | path |
| `taylorGreen` | `taylor_green` | path |
| `cylinderIBM` | `cylinder_ibm` | path |
| `coupled_FVM_VPM` | `coupled_fvm_vpm` | path |
| `naca4412Flow` | `naca4412_flow` | path |
| `cylinderSheddingFlow` | `cylinder_shedding_flow` | path |

## 14. Spelling normalisation

| Old | New | Scope |
|---|---|---|
| `center` | `centre` | all |
| mixed casing leftovers | `neighbour` | all |

## Mechanical verification (PR9 sweep)

After the refactor, the following must return no hits in `source/`, `openonda/`,
`tutorials/`, and `tests/` (except where documented in this manifest):

- `n_elements`, `element_centroid`, `element_volume`, `elem_`
- `self\.U\b`, `self\.p\b`, `self\.phi\b`, `self\.nut\b` **restricted to
  `source/solvers/FVM/`**.  `self.p` in the VLM/panel Krylov solvers is the
  CG/BiCGSTAB search direction, not pressure, and stays as is.
- `circulation`, `circulation_cpu`, `strengths`, `gamma`
- `setup_fvm_solver`, `setup_vpm_solver`, `setup_coupler`
- `backup_frequency`, `backup_file_name`, `BackupSystem`
- `delta_t`, `flow_time`
- bare `dt` / `*_dt` / `dt_*` as a time-step size.  Exempt, by category:
  - NumPy/Taichi `*dtype*`;
  - derivative notation `d(x)/dt`: `ddt_scheme`, `ddt_flux_correction`, `dE_dt`,
    `dU_dt_peak`, `du_dt`, `dalpha_dt`, `dstr_dt`;
  - `dT`, `dT_arr`, `dT_dr` — thrust/temperature increments, not time;
  - `four_nu_dt` — names the product 4·nu·Δt, not a step size;
  - `OPENONDA_FVM_DT` / `OPENONDA_VPM_DT` — external env-var contract, kept as
    string literals while the Python constants become `FVM_TIME_STEP_SIZE` /
    `VPM_TIME_STEP_SIZE`;
  - archived dataset/case names on disk (`relaxed_reference_*dt002*`).
- `E_previous`, `E_previous2`
- `alpha_u`, `alpha_p`, `momentum_maxiter`, `amg_maxiter`, `_tol`
- `\bcenter\b` in code (excluding third-party strings)
- stray `startFace`, `nFaces`, `neighbourPatch` outside mesh-import adapters
