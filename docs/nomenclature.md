# OpenONDA nomenclature and naming contract

This document is the canonical naming contract for OpenONDA. It defines a single
vocabulary for every concept that appears at more than one call site. All new
code must follow it, and the ongoing nomenclature refactor brings existing code
in line with it.

The companion file `docs/rename-manifest.md` records the concrete
old-name → new-name mapping and its scope. Nothing here changes numerical
behaviour: renames are purely semantic.

## Guiding rules

1. **Names must read as engineering physics, not computer science.** Prefer
   `kinematic_viscosity` over `nu` outside a small numerical kernel; prefer
   `velocity` over `U` at any API boundary.
2. **Descriptive names at the public/API level.** Single-letter equation symbols
   (`u`, `p`, `phi`, `rho`, `nu`, `omega`, `gamma`, `Sf`, `Co`) are
   allowed only inside small numerical kernels where the surrounding physics is
   unambiguous. They must never appear in the public `openonda` facade, in
   tutorial/script variables, or in serialized names.
3. **`Setup` for the top level, `Config` for subsystems.**
   - `FVMSetup`, `VPMSetup`, `CouplerSetup` are the complete, top-level
     configuration objects a user builds.
   - `Config` names only subsystem-scoped configurations
     (`TimeConfig`, `SchemesConfig`, `TurbulenceConfig`, …).
4. **Constructor arguments hold setups; attributes hold state.** A constructor
   that receives a `*Setup` must call its parameter `setup` (never `config`).
   The solver stores it immutably as `self.setup`. Everything that evolves in
   time lives in `self.state`. Never mutate `self.setup` at runtime.
5. **Names state units or cadence explicitly.** Anything that can be mistaken
   between "number of steps" and "seconds" must say so: `output_interval_steps`,
   `sampling_interval_time`, `checkpoint_interval_steps`. Never use `frequency`
   to mean "every N steps".
6. **British/OpenFOAM spelling is canonical:** `neighbour`, `centre`. Mixed
   `center`/`centre` spellings must be eliminated in favour of the British form.
7. **Abbreviations are whitelisted.** Acronyms that are standard in CFD keep
   their casing; ad-hoc abbreviations must be expanded. See
   [Abbreviations](#abbreviations) below.
8. **Never overload one identifier with different physical meanings.** For
   example, `alpha_p` means pressure relaxation in FVM and particle strength in
   VPM; both are renamed (`pressure_relaxation` and `vortex_strength`).
9. **Serialized names equal live names.** Checkpoint files, CSVs, VTK/HDF5
   datasets and the in-memory attributes they represent must use the same
   spelling.

## Canonical vocabularies

### Time

| Concept | Canonical name | Units | Notes |
|---|---|---|---|
| Time-step size | `time_step_size` | s | Property of a solver or its time config. Never abbreviated to `dt`, not even in kernels |
| Current simulation time | `time` | s | Runtime state, not setup |
| Time-step index | `step` | – | Runtime state, not setup |
| Start of integration | `start_time` | s | |
| End of integration | `end_time` | s | |
| Number of steps | `n_steps` | – | |
| Substeps within a step | `substep_size` / `n_substeps` | s / – | e.g. coupler sub-cycling |
| Solver-specific step size | `fvm_time_step_size` / `vpm_time_step_size` | s | Coupler context |
| Sampling cadence in steps | `sampling_interval_steps` | – | |
| Sampling cadence in time | `sampling_interval_time` | s | |
| Output cadence | `output_interval_steps` | – | |
| Logging cadence | `logging_interval_steps` | – | |

### Fluid properties

| Concept | Canonical name | Units |
|---|---|---|
| Density | `density` | kg/m³ |
| Kinematic viscosity | `kinematic_viscosity` | m²/s |
| Dynamic viscosity | `dynamic_viscosity` | Pa·s |
| Turbulent (eddy) viscosity | `eddy_viscosity` | m²/s |
| Effective viscosity (molecular + eddy) | `effective_viscosity` | m²/s |
| Kinematic pressure (p/ρ) | `kinematic_pressure` | m²/s² |
| Pressure | `pressure` | Pa |

### FVM fields

| Old | New | Units | Meaning |
|---|---|---|---|
| `U` | `velocity` | m/s | Cell velocity vector |
| `p` | `kinematic_pressure` | m²/s² | p/ρ — NOT Pa |
| `phi` | `face_flux` | m³/s | Face volumetric flux |
| `U_old` / `U_old_old` | `velocity_old` / `velocity_older` | m/s | Previous time levels |
| `phi_old` / `phi_old_old` | `face_flux_old` / `face_flux_older` | m³/s | Previous time levels |
| `nut` | `eddy_viscosity` | m²/s | |
| `initial_p` | `initial_kinematic_pressure` | m²/s² | |

The same vocabulary is used for `FieldState` and any field container.

### Mesh

| Old | New | Notes |
|---|---|---|
| `n_elements` | `n_cells` | |
| `element_centroids` | `cell_centroids` | |
| `element_volumes` | `cell_volumes` | |
| `element_faces` | `cell_faces` | |
| `element_ids` | `cell_ids` | |
| `elem_*` | `cell_*` | any remaining prefix |
| `startFace` | `start_face` | raw boundary key, normalized at the mesh-import adapter only |
| `nFaces` | `n_faces` | raw boundary key, normalized at the mesh-import adapter only |
| `neighbourPatch` | `neighbour_patch` | raw boundary key, normalized at the mesh-import adapter only |

### Boundary conditions

| Old | New |
|---|---|
| `type_velocity` | `velocity_type` |
| `value_velocity` | `velocity_value` |
| `type_p` | `pressure_type` |
| `value_p` | `kinematic_pressure_value` |
| `type_phi` | `flux_type` |
| `type_nut` | `eddy_viscosity_type` |

`BoundaryConfig` factory constructors keep their names: `inlet`, `outlet`,
`wall`, `freestream`.

### Configuration classes

| Old | New | Notes |
|---|---|---|
| `MeshConfig` | `MeshQualityConfig` | disambiguation from `MeshSetup`-style objects |
| `SchemesConfig` | `DiscretizationConfig` | |
| `ExecutionConfig` | `ComputeConfig` | |
| `OutputSetup` | `OutputConfig` | |
| `LogConfig` | `LoggingConfig` | |
| `DynamicMeshConfig` | `MeshMotionConfig` | |

Numerical controls:

| Old | New |
|---|---|
| `alpha_u` | `velocity_relaxation` |
| `alpha_p` | `pressure_relaxation` |
| `momentum_maxiter` | `momentum_max_iterations` |
| `amg_maxiter` | `amg_max_iterations` |
| `*_tol` | `*_tolerance` |

LES coefficients (do not overload one field with several physical meanings):

| Old | New | Meaning |
|---|---|---|
| `Cs` | `c_s` | Smagorinsky constant |
| `Ck` | `c_k` | Kinetic-energy coefficient |
| `Ce` | `c_e` | Dissipation coefficient |
| `Cw` | `c_w` | Wall coefficient |

### VPM particles

The live particle vortex field is a **3D vector**, `α_p ≈ ω_p·V_p` (units
L³/T). It is NOT the scalar circulation `Γ` (units L²/T), and it must never be
named `circulation`.

| Old | New | Units | Meaning |
|---|---|---|---|
| `circulation` (field) | `vortex_strength` | m³/s | α_p = ω_p V_p |
| `circulation_cpu()` | `vortex_strength_cpu()` | m³/s | |
| `position` | `position` | m | already canonical |
| `velocity` | `velocity` | m/s | already canonical |
| `radius` | `core_radius` | m | Gaussian core radius |
| `viscosity` | `kinematic_viscosity` | m²/s | molecular |
| `viscosity_turbulent` | `eddy_viscosity` | m²/s | |
| `viscosity_effective` | `effective_viscosity` | m²/s | |
| `grad_u` | `velocity_gradient` | 1/s | |
| `group_id` | `group_id` | – | already canonical |
| `vorticity` | `vorticity` | 1/s | already canonical |

Serialized names (HDF5 datasets, VTK point data, `ParticlesState` model, JSON)
must match the live names exactly, in singular form.

### VPM solver and setup

| Old | New | Notes |
|---|---|---|
| `Solver` | `VPMSolver` | |
| `time_step_size` | `time_step_size` | unchanged; already canonical |
| `characteristic_distance` | `particle_spacing` | |
| `particles_kernel` | `particle_kernel` | |
| `processing_unit` | `compute_device` | |
| `max_targets` | `max_evaluation_points` | |
| `vpm_domain_bounds` | `domain_bounds` | |
| `regen_*` | `regeneration_*` | |
| `dvh_rd_ratio` | `dvh_support_radius_ratio` | |
| `SetFlowModel` | `set_flow_model` | |
| `CachedParticleProperty` | `cached_particle_property` | |
| `self.LES` | `self.turbulence_model` | |
| `number_of_particles` | `n_particles` | |
| `num_sources` | `n_sources` | |
| `E_previous` / `E_previous2` | (removed) | dead fields; see manifest |

### Coupler

| Old | New |
|---|---|
| `FVMVPMCoupler` constructor order `(vpm_solver, fvm_solver, setup)` | `(fvm_solver, vpm_solver, coupler_setup)` |
| `self.fvm` | `self.fvm_solver` |
| `self.vpm` | `self.vpm_solver` |
| `self.transfer` | `self.vorticity_transfer` |
| `self.blending` | `self.blending_zone` |
| `self.t_end` | `self.end_time` |
| `self.dt_fvm` / `self.dt_vpm` | `self.fvm_time_step_size` / `self.vpm_time_step_size` |
| `self.nu` / `self.rho` | `self.kinematic_viscosity` / `self.density` |
| `n_fvm_substeps` | `fvm_substeps` |
| `_u_bc_prev` (and similar) | `_previous_boundary_velocity` (etc.) |

`CouplerSetup` fields:

| Old | New |
|---|---|
| `transfer_region_box` | `transfer_region_bounds` |
| `bc_patch_name` | `coupling_patch` |
| `vpm_bc_mode` | `boundary_condition_mode` |
| `transfer_prune_vorticity_min` | `transfer_vorticity_cutoff` |
| `coupler_backup_period` | `checkpoint_interval_steps` |
| `overlap_zone_dead_zone_width` | `vpm_only_width` |
| `overlap_zone_ramp_width` | `authority_ramp_width` |

`CouplerSetup` fields are grouped into at most one level of namespacing:
`overlap`, `transfer`, `boundary`, `pressure_reference`, `checkpoint`.

### Checkpoint / restart

Use `checkpoint` vocabulary, not `backup`.

| Old | New |
|---|---|
| `backup_frequency` | `checkpoint_interval_steps` |
| `backup_file_name` | `checkpoint_name` |
| `BackupSystem` | `CheckpointManager` / `CheckpointIO` |

### Sampling

- A `SamplingSchedule` object is the single vocabulary for when samples are
  taken (`sampling_interval_steps` / `sampling_interval_time`).
- Sampler classes keep their names: `LineSampler`, `SurfaceSampler`,
  `ForceSampler` (prefixed `FVM…` only when the sampler is solver-specific).
- Tutorial sampling variables: `sample_spacing`, `sample_interval_time`,
  `sample_interval_steps`, `sample_bounds`.

## Stepping API

| Old | New | Semantics |
|---|---|---|
| `evolve()` (FVM) | `advance()` | One physical time step |
| `update_state()` (VPM) | `advance()` | One physical time step |
| `run()` | `run()` | Advance until an end condition |
| `solve_*()` | `solve_*()` | Solve the current time level without advancing |

## Tutorials and scripts

- Import through the `openonda.fvm`, `openonda.vpm`, `openonda.coupler`
  namespaces. Do not alias third-party or internal names
  (`LineSampler as FVMLineSampler`, `Solver as FVM_Solver`).
- Build functions: `create_fvm_setup()`, `create_vpm_setup()` (was
  `build_config()`, `make_vpm_setup()`).
- Constants use descriptive, unit-honest names:

| Old | New | Units |
|---|---|---|
| `RHO` | `DENSITY` | kg/m³ |
| `NU` | `KINEMATIC_VISCOSITY` | m²/s |
| `U_INF` | `FREESTREAM_VELOCITY` (vector) | m/s |
| `U_INF_MAG` | `FREESTREAM_SPEED` (scalar) | m/s |
| `L_CUBE` | `CUBE_SIDE` | m |
| `DX` / `H` | `CELL_SIZE` | m |
| `DS` / `H_PARTICLES` | `PARTICLE_SPACING` | m |
| `N_MAX` | `MAX_PARTICLES` | – |
| `T_END` | `END_TIME` | s |

- Variable names: `fvm_setup`, `vpm_setup`, `coupler_setup`, `fvm_solver`,
  `vpm_solver`, `coupled_solver`.
- Tutorial directory/file names are `snake_case`, keeping preserved
  nomenclature tokens (NACA4412, VPM, FVM, IBM): `cube_flow`,
  `boundary_layer`, `lamb_oseen_vortex`, `taylor_green`, `cylinder_ibm`,
  `coupled_fvm_vpm`, `naca4412_flow`, `cylinder_shedding_flow`.

## Abbreviations

Allowed broadly (standard CFD/acronyms, keep casing):

`FVM`, `VPM`, `VLM`, `LES`, `DNS`, `IBM`, `PIMPLE`, `PISO`, `SIMPLE`, `CFL`,
`AMG`, `ILU`, `RK`, `DVH`, `GBD`, `RWM`, `STL`, `VTK`, `MPI`, `GPU`, `CPU`.

Allowed only in tight equation-local scope:

`u`, `p`, `phi`, `rho`, `nu`, `mu`, `omega`, `gamma`, `Sf`, `Co`.

`dt` is deliberately **not** on this list: the time-step size is spelled
`time_step_size` in every scope, including numerical kernels.

Expanded everywhere else (avoid in large scopes):

`cfg` → `config`/`setup`, `arr` → `array`, `val` → `value`, `obj` → `object`,
`tmp`/`temp` → descriptive, `res` → `result`/`residual`, `psys` → `particles`,
`vsc` → `viscous`, `vc` → `velocity`, `dom` → `domain`, `bg` → `background`,
`max_p` → `max_particles`.

## What this refactor is NOT

- No numerical changes: no equation rewrites, no loop-order changes, no
  precision changes, no tolerance changes, no interpolation/transfer-formula
  changes, no array-layout changes, no unrelated bug fixes.
- Renames are atomic with their call sites: internal references, tests,
  tutorials, scripts and the `openonda` facade are updated in the same change.
- No backward-compatibility aliases. Hard renames are the contract.