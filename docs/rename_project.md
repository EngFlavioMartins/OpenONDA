Backups, checkpoints, archived runs, generated solver outputs, post-processing readers, tests, tutorials, and file/package names are all included. This document is the completed audit record for the one-way rename.

# Master TODO: physical-variable nomenclature

## 1. Global rules to freeze first

- [x] Keep canonical physical names in the solver and serializer code that owns them; do not maintain a duplicate runtime field registry.
- [x] Restore `docs/nomenclature.md`.
- [x] Create `docs/rename_manifest.md` containing every old → canonical mapping.
- [x] Require lower `snake_case` for APIs, state attributes, dictionary keys, NPZ/HDF5 datasets, XDMF/VTK arrays, CSV columns, JSON/JSONL keys, and logs.
- [x] Use the same name for the same physical quantity in every solver.
- [x] Prohibit the same name from representing quantities with different units or tensor ranks.
- [x] Document field meaning, SI units, shape, components, and centering beside the owning state or serializer.
- [x] Use singular field names for arrays: `velocity`, `vorticity`, `position`, not `velocities`.
- [x] Flatten vector components as `<field>_x`, `<field>_y`, `<field>_z`.
- [x] Flatten tensors as `<field>_xx`, `<field>_xy`, etc.
- [x] Use `n_*` for counts: `n_particles`, `n_panels`, `n_fvm_substeps`.
- [x] Use `is_*` and `has_*` for booleans.
- [x] Use explicit cadence suffixes: `_interval_steps`, `_interval_time`.
- [x] Use `_magnitude`, `_mean`, `_min`, `_max`, `_rate`, `_ratio`, `_percent`, `_degrees`, `_radians`, `_l1`, `_l2`, and `_linf` consistently.
- [x] Allow compact mathematical symbols only inside small numerical kernels and equations; never use them as physical API, state, or output names.
- [x] Never expose `U`, `p`, `phi`, `nu`, `rho`, `omega`, `gamma`, `dt`, `Co`, or `nut` through an API or output.
- [x] Prohibit external solver spellings such as `U`, `p`, `phi`, and `nut` even inside import/export adapters; translate at the boundary.
- [x] Remove old names completely; do not add migration aliases or legacy readers.
- [x] Make each writer emit the canonical names used by its owning solver and enforce them with the static nomenclature gate.
- [x] Give every restart format its own local format version; do not add a second global physical-field schema version.

## 2. Canonical physical names

### Time and execution

| Remove | Canonical |
|---|---|
| `flow_time`, `t` outside kernels | `time` |
| `time_step` when it means an index | `step` |
| `dt`, `delta_t`, `time_step` when it means seconds | `time_step_size` |
| `observed_dt` | `observed_time_step_size` |
| `fvm_substeps`, `n_fvm_substeps` mixture | `n_fvm_substeps` |
| `cfl`, `Co` | `courant_number` |
| `cfl_max`, `max_cfl` | `max_courant_number` |
| `backup_frequency` | `checkpoint_interval_steps` |
| `BACKUP_*` used for ordinary output | `OUTPUT_*` |
| `BACKUP_*` used for restart state | `CHECKPOINT_*` |

The FVM distinction between `time_step_size` and `_current_time_step_size` must be made semantic:

- `time_step_size`: adaptive candidate for the next/current advance.
- `accepted_time_step_size`: size of the last accepted physical step.

### Coordinates and geometry

| Remove | Canonical |
|---|---|
| bare CSV `x`, `y`, `z` | `position_x`, `position_y`, `position_z` |
| `center`, `centre`, `centroid` used interchangeably | meaning-specific names below |
| cell geometric location | `cell_centre` |
| face geometric location | `face_centre` |
| panel geometric location | `panel_centre` |
| weighted vortex location | `vortex_centroid` |
| `PanelCenter` | `panel_centre` |
| `PanelChord` | `panel_chord` |
| `BoundLeg` | `bound_vortex_leg` |
| `Normal` | `normal` |
| `Area` | `area` |
| `IsTE`, `IsLE` | `is_trailing_edge`, `is_leading_edge` |

Use the project-wide British spellings `centre` and `neighbour`. Use `centroid` only for a mathematically weighted centroid.

### Velocity and derivatives

| Remove | Canonical |
|---|---|
| `U`, `V`, `Velocity` | `velocity` |
| `Ux`, `Uy`, `Uz` | `velocity_x`, `velocity_y`, `velocity_z` |
| `V_inf`, `U_inf` | `freestream_velocity` |
| `U_inf_mag` | `freestream_speed` |
| `grad_u`, `VelocityGradient` | `velocity_gradient` |
| `dudx` | `velocity_gradient_xx` |
| `dudy` | `velocity_gradient_xy` |
| `dudz` | `velocity_gradient_xz` |
| `dvdx` … `dwdz` | corresponding `velocity_gradient_yx` … `velocity_gradient_zz` |
| `Sxx`, `Sxy`, etc. | `strain_rate_xx`, `strain_rate_xy`, etc. |
| `StrainRate` | `strain_rate` |
| `VelocityMagnitude` | `velocity_magnitude` |
| `BoundVelocity` | `bound_vortex_velocity` |

For point-evaluation APIs:

| Current style | Canonical API |
|---|---|
| `compute_target_velocities` | `compute_velocity_at_points` |
| `compute_target_velocity_gradients` | `compute_velocity_gradient_at_points` |
| `compute_complete_target_velocity_and_gradients` | `compute_velocity_and_gradient_at_points` |

### Pressure and flux

| Remove | Canonical |
|---|---|
| FVM `p`, `Pressure` | `kinematic_pressure` |
| true pressure in Pa | `pressure` |
| `phi`, `face_flux` | `volumetric_face_flux` |
| future mass flux | `mass_flux` |
| `Cp` | `pressure_coefficient` |
| `DeltaCp` | `pressure_jump_coefficient` |
| `PressureGradient` if kinematic | `kinematic_pressure_gradient` |

History fields become:

- `velocity`
- `velocity_old`
- `velocity_older`
- `volumetric_face_flux`
- `volumetric_face_flux_old`
- `volumetric_face_flux_older`

### Viscosity and turbulence

| Remove | Canonical |
|---|---|
| `nu`, `viscosity` | `kinematic_viscosity` |
| `nut`, `turbulent_viscosity`, `viscosity_turbulent` | `eddy_viscosity` |
| `viscosity_effective` | `effective_viscosity` |
| `mu` outside kernels | `dynamic_viscosity` |
| `Cs` | `smagorinsky_coefficient` |
| `Cw` | `wale_coefficient` |
| ambiguous `effective_diffusivity` | keep only if it is genuinely diffusivity rather than viscosity |

The owning solver documentation must state that `eddy_viscosity` and `effective_viscosity` are kinematic quantities in m²/s.

### Vorticity, circulation, and VPM strength

This distinction is mandatory:

- `vorticity`: \(\omega\), units 1/s.
- `vortex_strength`: VPM vector \(\alpha_p \approx \omega_p V_p\), units m³/s.
- `circulation`: true scalar \(\Gamma\), units m²/s, used by VLM and ring physics only.

| Remove | Canonical |
|---|---|
| `omega`, `Vorticity` | `vorticity` |
| `omega_x`, etc. | `vorticity_x`, etc. |
| VPM `circulation`, `gamma`, `strength` | `vortex_strength` |
| VPM `max_gamma` | `max_vortex_strength_magnitude` |
| particle `radius` | `core_radius` |
| `VortexStrength` | `vortex_strength` |
| `CoreRadius` | `core_radius` |
| `GroupID`, `ZoneID` | `group_id`, `zone_id` |
| particle `vector_circulation` | `net_vortex_strength` |
| `vector_circulation_x` | `net_vortex_strength_x` |
| `length_strength` | `vortex_strength_magnitude_sum` |
| `strength_magnitude` | `vortex_strength_magnitude_sum` |
| `strength_x/y/z` when they are vector sums | `net_vortex_strength_x/y/z` |

The following three quantities must remain distinct:

- `vortex_strength_magnitude_sum` = \(\sum_p |\alpha_p|\)
- `net_vortex_strength` = \(\sum_p \alpha_p\)
- `net_vortex_strength_magnitude` = \(|\sum_p \alpha_p|\)

True VLM/ring names that remain valid:

- `circulation`
- `bound_circulation`
- `wake_circulation`
- `tube_circulation`

### Forces and moments

| Remove | Canonical |
|---|---|
| `Fx`, `Fy`, `Fz` | `force_x`, `force_y`, `force_z` |
| `Fpx`, etc. | `pressure_force_x`, etc. |
| `Fvx`, etc. | `viscous_force_x`, etc. |
| `Ftx`, etc. | `total_force_x`, etc. |
| `Mx`, `My`, `Mz` | `moment_x`, `moment_y`, `moment_z` |
| `L`, `D`, `C` | `lift`, `drag`, `side_force` |
| `CL`, `Cd`, `CD` | `lift_coefficient`, `drag_coefficient` |
| `CC`, `Cz` | `side_force_coefficient` |
| `Cl` | `rolling_moment_coefficient` |
| `Cm` | `pitching_moment_coefficient` |
| `Cn` | `yawing_moment_coefficient` |
| `q` | `dynamic_pressure` |
| `S_ref` | `reference_area` |
| `c_ref` | `reference_chord` |
| `b_ref` | `reference_span` |
| `r_ref` | `reference_point` |
| `PanelForce` | `panel_force` |
| `DoubletStrength` | `doublet_strength` |

### Integral diagnostics

| Remove | Canonical |
|---|---|
| `kinetic_energy` for a global integral | `total_kinetic_energy` |
| `enstrophy` for a global integral | `total_enstrophy` |
| `enstrophy_test` | `test_filtered_enstrophy` |
| `helicity` for a global integral | `total_helicity` |
| `dEdt` | `kinetic_energy_rate` |
| `neg_nu_enstrophy` | `viscous_kinetic_energy_rate` |
| `impulse_x` | `linear_impulse_x` |
| `angle_rad` | `angle_radians` |
| `strength_misalignment_deg` | `vortex_strength_misalignment_degrees` |
| `stabilization_circulation_error` | `stabilization_vortex_strength_error` |
| `stabilization_strength_growth` | `stabilization_vortex_strength_growth` |
| `max_strength` | `max_vortex_strength_magnitude` |

`kinetic_energy_rate` is signed \(dE/dt\). A quantity called `dissipation_rate` must be nonnegative by definition.

## 3. Backup and checkpoint audit

Backups are a dedicated workstream, not a side effect of source renaming.

### Serial FVM checkpoint

Files:

- [checkpoint.py](/Users/flaviomartins/OpenONDA/source/solvers/fvm/io/checkpoint.py)
- FVM restart tests and fixtures.

Tasks:

- [x] Bump checkpoint version to 7.
- [x] Rename `face_flux*` to `volumetric_face_flux*`.
- [x] Confirm `time_step_size` and `accepted_time_step_size` state names.
- [x] Use the checkpoint-local `format_version` as the complete serialization contract.
- [x] Validate exact required and unexpected keys.
- [x] Reject old checkpoint formats; no legacy migration path.
- [x] Test write → inspect → load equality.

### Partitioned/MPI FVM checkpoint

Files:

- [partitioned.py](/Users/flaviomartins/OpenONDA/source/solvers/fvm/io/partitioned.py)
- Partitioned-state contract validation.

Tasks:

- [x] Replace `dt` with `time_step_size`.
- [x] Use exactly the same physical keys as serial FVM.
- [x] Rename flux fields consistently.
- [x] Bump partitioned format version to 5.
- [x] Update and strictly validate manifest metadata.
- [x] Reject old checkpoint formats; no legacy migration path.
- [x] Validate partitioned rank payloads and reconstructed state against the same exact-key contract.

### VPM checkpoint

Files:

- [checkpoint.py](/Users/flaviomartins/OpenONDA/source/solvers/vpm/io/checkpoint.py)
- [state.py](/Users/flaviomartins/OpenONDA/source/solvers/vpm/config/state.py)
- [offline.py](/Users/flaviomartins/OpenONDA/source/solvers/vpm/diagnostics/offline.py)

Formats:

- HDF5
- XDMF
- JSON setup/configuration

Tasks:

- [x] Bump VPM checkpoint version to 7.0.
- [x] Make HDF5 dataset and XDMF Attribute names identical.
- [x] Replace PascalCase XDMF names with canonical snake case.
- [x] Remove heuristic old-name lookup from normal readers.
- [x] Reject old checkpoint formats; no legacy migration path.
- [x] Cover zero-particle checkpoints in the serializer contract.
- [x] Validate optional stabilization/reference fields explicitly.
- [x] Preserve the solver's configured numeric dtype on load.

### Coupled checkpoint

Files:

- [checkpoint.py](/Users/flaviomartins/OpenONDA/source/coupler/checkpoint.py)
- [solver.py](/Users/flaviomartins/OpenONDA/source/coupler/solver.py)
- [boundary.py](/Users/flaviomartins/OpenONDA/source/coupler/boundary.py)

Tasks:

- [x] Bump coupled format version to 9.
- [x] Validate each child artifact through its own checkpoint-local format version.
- [x] Replace `vpm_bc` with `vpm_boundary_condition_state`.
- [x] Rename boundary arrays to their full live-state names.
- [x] Use `n_fvm_substeps` consistently.
- [x] Validate nested artifact formats before loading.
- [x] Reject old coupled checkpoint formats; no legacy migration path.
- [x] Test the complete boundary history and coupled child-artifact round trip.

### Other recoverable state

Files:

- [container.py](/Users/flaviomartins/OpenONDA/source/solvers/vpm/particles/container.py)
- VPM VTP particle import/export.
- FVM offline snapshot post-processing.
- VLM/panel surface output if used for restart or continuation.

Tasks:

- [x] Keep particle VTP visualization fields canonical without a duplicate global schema stamp.
- [x] Rename VTK particle arrays.
- [x] Require current canonical particle fields; do not add versioned migrations.
- [x] Separate visualization-only files from restart-qualified files.

### Existing backup/archive directories

Explicitly include:

- `tutorials/coupled_fvm_vpm/cube_flow/reference_flow/**`
- `tutorials/**/solution/*.h5`
- `tutorials/**/solution/*.xdmf`
- any NPZ/HDF5/VTP checkpoint outside `solution/`

Verified state:

- [x] Rename `run_backups` → `run_archives`.
- [x] Rename `samples_backup` → `samples_archive`.
- [x] Preserve tracked cube reference data in its canonical archive directories.
- [x] Scan archive text plus NPZ/HDF5 field names without rewriting historical results in place.
- [x] Keep restart-qualified state under `solution/checkpoints/`, distinct from samples and visualization output.

## 4. File-focused audit

### Shared/public packages

- [x] `openonda/fvm.py`
- [x] `openonda/vpm.py`
- [x] `openonda/coupler.py`
- [x] `openonda/__init__.py`
- [x] `source/solvers/__init__.py`
- [x] `source/solvers/fvm/__init__.py`
- [x] `source/solvers/vpm/__init__.py`
- [x] FVM/VPM factories and capabilities JSON.

Python package directories are lowercase after the one-way rename:

- `source/solvers/FVM` → `source/solvers/fvm`
- `source/solvers/VPM` → `source/solvers/vpm`

Class names remain `FVMSolver`, `VPMSolver`, `FVMVPMCoupler`.

### FVM live state and numerics

Audit and update all Python files under:

- [x] `source/solvers/fvm/core/`
- [x] `source/solvers/fvm/config/`
- [x] `source/solvers/fvm/assemble/`
- [x] `source/solvers/fvm/solve/`
- [x] `source/solvers/fvm/fields/`
- [x] `source/solvers/fvm/schemes/`
- [x] `source/solvers/fvm/turbulence/`
- [x] `source/solvers/fvm/immersed_boundary/`
- [x] `source/solvers/fvm/coupling/`
- [x] `source/solvers/fvm/mesh/`

Priority files:

- [state.py](/Users/flaviomartins/OpenONDA/source/solvers/fvm/core/state.py)
- [solver.py](/Users/flaviomartins/OpenONDA/source/solvers/fvm/core/solver.py)
- [types.py](/Users/flaviomartins/OpenONDA/source/solvers/fvm/config/types.py)
- `simple_solver.py`
- `pimple_solver.py`
- `momentum.py`
- `boundaries.py`
- `diagnostics.py`

### FVM outputs

- [x] `io/checkpoint.py`
- [x] `io/partitioned.py`
- [x] `io/vtk_exporter.py`
- [x] `io/async_output.py`
- [x] `io/solver_io.py`
- [x] `io/profiling.py`
- [x] `io/manifest.py`
- [x] `io/logging.py`
- [x] `sampling/base.py`
- [x] `sampling/executor.py`
- [x] `sampling/fields.py`
- [x] `sampling/forces.py`
- [x] `sampling/postprocess.py`

### VPM live state and numerics

Audit every Python file under:

- [x] `source/solvers/vpm/config/`
- [x] `source/solvers/vpm/core/`
- [x] `source/solvers/vpm/particles/`
- [x] `source/solvers/vpm/physics/`
- [x] `source/solvers/vpm/stabilization/`
- [x] `source/solvers/vpm/turbulence/`
- [x] `source/solvers/vpm/diagnostics/`
- [x] `source/solvers/vpm/initial_conditions/`
- [x] `source/solvers/vpm/acceleration/`
- [x] `source/solvers/vpm/kernels/`
- [x] `source/solvers/vpm/numerics/`
- [x] `source/solvers/vpm/coupling/`

Priority files:

- `core/solver.py`
- `core/evolution.py`
- `particles/container.py`
- `particles/distribution.py`
- `physics/evaluation.py`
- `physics/pressure.py`
- `physics/diffusion/*.py`
- `stabilization/manager.py`
- `stabilization/regularization.py`

### VPM outputs

- [x] `io/checkpoint.py`
- [x] `io/vtk_export.py`
- [x] `io/csv_export.py`
- [x] `io/monitor.py`
- [x] `io/runtime_profiler.py`
- [x] `io/sampler.py`
- [x] `io/solver_io.py`
- [x] `io/sampling/field_samplers.py`
- [x] `diagnostics/conservation.py`
- [x] `diagnostics/offline.py`
- [x] `diagnostics/ring.py`

### VLM and panel solvers

Audit all files under:

- [x] `source/solvers/vpm/boundary_elements/vlm/`
- [x] `source/solvers/vpm/boundary_elements/panels/`

Priority files:

- `vlm/solver/lattice.py`
- `vlm/solver/vlm_solver.py`
- `vlm/solver/forces.py`
- `vlm/solver/diagnostics.py`
- `vlm/solver/loading_distribution.py`
- `panels/solver/panel_solver.py`
- `panels/solver/diagnostics.py`
- `panels/solver/forces.py`
- `panels/solver/vtk_export.py`
- panel/VLM surface metadata JSON writers.

### Coupler

- [x] `source/coupler/solver.py`
- [x] `source/coupler/checkpoint.py`
- [x] `source/coupler/boundary.py`
- [x] `source/coupler/vorticity_transfer.py`
- [x] `source/coupler/interpolation.py`
- [x] `source/coupler/pressure_reference.py`
- [x] `source/coupler/reporting.py`
- [x] `source/coupler/config/types.py`
- [x] FVM coupling interface.

Rename abbreviated boundary state such as `_velocity_bc_prev` to full names such as `_previous_boundary_velocity`.

### Tests

- [x] All `tests/fvm/**/*.py`
- [x] All `tests/vpm/**/*.py`
- [x] All `tests/coupler/**/*.py`
- [x] `tests/test_public_api_has_no_legacy_aliases.py`
- [x] `scripts/validate_native_tutorials.py` and tutorial setup-contract tests.
- [x] Checkpoint fixtures and generated golden files.
- [x] Tests with physics-local `U`, `p`, `rho`, `nu`, `gamma`, or `dt`.
- [x] Tests whose names still use `backup`, `flow_time`, or `time_step`.

Migration fixtures are intentionally unsupported; old formats must be rejected instead.

### Tutorials, scripts, and documentation

- [x] All tutorial setup files.
- [x] All plotting and validation assets.
- [x] All benchmark scripts.
- [x] All experiment scripts.
- [x] Shell scripts that reference renamed files/directories.
- [x] README files.
- [x] ParaView `.pvsm` state files.
- [x] Versioned CSV, JSON, VTS, VTU, VTP, PVD, PVTU, HDF5, XDMF, and NPZ files.

Verified one-way path mappings:

- `lambOseenVortex` → `lamb_oseen_vortex`
- `vortexRing` → `vortex_ring`
- `vortexInteractions` → `vortex_interactions`
- `quadCopter` → `quadcopter`
- `rotorFlow` → `rotor_flow`
- `deltaWing` → `delta_wing`
- `flatPlate` → `flat_plate`
- `coupled_FVM_VPM` → `coupled_fvm_vpm`
- `cubeFlow_setup.py` → `cube_flow_setup.py`
- `referenceFlow_setup.py` → `reference_flow_setup.py`
- `cylinderSheddingFlow_setup.py` → `cylinder_shedding_flow_setup.py`

## 5. Execution order

- [x] **Stage 1 — Contract:** nomenclature document, full rename manifest, and owning-code contracts.
- [x] **Stage 2 — Enforcement:** AST, serializer, path, and generated-artifact checks.
- [x] **Stage 3 — Live state:** FVM, VPM, coupler, VLM, and panel runtime/API names.
- [x] **Stage 4 — Checkpoints:** all restart formats locally versioned; old formats rejected.
- [x] **Stage 5 — Outputs:** canonical VTK/XDMF/HDF5/CSV/JSON/JSONL field contracts.
- [x] **Stage 6 — Consumers:** post-processing, tutorials, scripts, tests, and ParaView states.
- [x] **Stage 7 — Filesystem:** lowercase packages, tutorial directories, filenames, shell paths, and documentation links.
- [x] **Stage 8 — Stored data:** preserve the cube reference-flow data in canonical archive paths and scan it read-only.
- [x] **Stage 9 — Final gate:** prove zero unclassified legacy names.

## 6. Final “never repeat this” acceptance gate

The task is finished only when:

- [x] Every physical quantity has one canonical name in the owning solver and serializers.
- [x] Every physical quantity documents meaning, units, shape, components, and centering where it is owned.
- [x] Serial and MPI FVM checkpoints use identical field names.
- [x] HDF5 dataset names equal XDMF Attribute names.
- [x] Live attributes equal checkpoint keys where they represent the same state.
- [x] FVM and VPM samplers use identical names for common fields.
- [x] VLM `circulation` and VPM `vortex_strength` are never confused.
- [x] No new artifact contains PascalCase physical fields.
- [x] No CSV contains `Ux`, `omega_x`, `Sxx`, `dudx`, `p`, `dt`, `Fx`, `CL`, or `max_gamma`.
- [x] Old checkpoint formats are rejected; current checkpoints load with numerical equality.
- [x] All post-processors consume only canonical field names.
- [x] The maintained cube reference-flow artifacts pass the nomenclature scan.
- [x] External adapters translate physical names to the canonical contract; they receive no nomenclature exception.
- [x] Ruff, Pyrefly delta checks, the maintained unit/integration suite, and tutorial validation pass.
- [x] CI and report-only pre-commit hooks run the repository, path, and generated-artifact nomenclature gate.

## 7. Additional inconsistencies found during audit

- [x] Remove obsolete duplicate field-registry code instead of maintaining a second authority.
- [x] Remove the deleted cube timing helper and replace `fvm_dt`/`vpm_dt` in setup, environment, CLI, and tests.
- [x] Replace stale VPM `strength`, particle `radius`, and generic weighted-`centroid` APIs with meaning-specific names.
- [x] Remove the dead weak-particle pruning API that still called pre-rename particle methods.
- [x] Rename FVM geometric `centroid` locals to `cell_centre`, `face_centre`, or `geometry_centre`.
- [x] Add exact rejected-name coverage for the additional stale APIs discovered by this audit.
- [x] Add CI and report-only pre-commit enforcement so the rename cannot silently regress.
