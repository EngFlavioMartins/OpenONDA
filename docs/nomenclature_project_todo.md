# OpenONDA physical nomenclature project

Status: complete (2026-08-23)

Objective: one physics-oriented, Pythonic nomenclature across all solvers,
archives, checkpoints, tutorials, scripts, tests, and generated artifacts,
with no aliases or backward-compatibility paths.

## Global rules

- [x] Lower `snake_case` is mandatory at every API and serialization boundary.
- [x] One physical quantity has one name, unit, rank, component order, and
  location across FVM, VPM, VLM, panel, immersed-boundary, and coupled code.
- [x] Geometry uses British `centre` consistently; filenames use `centreline`.
- [x] Counts use `n_*`; booleans use `is_*` or `has_*`; time histories use
  `_old` and `_older`.
- [x] Vector and tensor component suffixes follow the canonical registry.
- [x] Compact symbols remain only inside isolated numerical kernels.
- [x] Readers accept only the current format and physical-field schema.
- [x] No public aliases, alternate keys, fallback spellings, migration
  readers, or dual-format writers remain.
- [x] Historical run copies are archives; restart state is stored only in
  checkpoint directories.

## Authoritative files

- [x] `source/schemas/physical_fields.py` defines every serialized physical
  field and its contract.
- [x] `source/schemas/serialization.py` stamps and validates field maps.
- [x] `docs/nomenclature.md` defines the human-readable semantics.
- [x] `docs/rename-manifest.md` records the final canonical filesystem and
  stored-artifact layout.
- [x] `scripts/check_nomenclature.py` scans source, paths, generated text,
  NPZ keys, and HDF5 object/attribute names.
- [x] Contributor and pre-commit guidance invokes the nomenclature gate.

## Canonical name families

- [x] Time: `time`, `step`, `time_step_size`,
  `accepted_time_step_size`, `observed_time_step_size`, `n_fvm_substeps`.
- [x] Geometry: `position`, `vertex_position`, `cell_centre`, `face_centre`,
  `panel_centre`, `rotation_centre`, `normal`, `face_area_vector`,
  `cell_volume`, `particle_volume`.
- [x] Velocity: `velocity`, `freestream_velocity`, `background_velocity`,
  `prescribed_velocity`, `kinematic_velocity`, `bound_vortex_velocity`,
  `velocity_magnitude`, `velocity_gradient`, `strain_rate`, `vorticity`.
- [x] Pressure/flux: `kinematic_pressure`, `pressure`,
  `kinematic_pressure_gradient`, `pressure_gradient`,
  `convective_pressure_gradient`, `temporal_pressure_gradient`,
  `viscous_pressure_gradient`, `volumetric_face_flux`, `mass_flux`,
  `courant_number`, `max_courant_number`.
- [x] Viscosity/turbulence: `kinematic_viscosity`, `dynamic_viscosity`,
  `eddy_viscosity`, `effective_viscosity`, and fully named model
  coefficients.
- [x] Vortex/VLM/panel: `vortex_strength`, `circulation`,
  `bound_circulation`, `wake_circulation`, `doublet_strength`,
  `bound_vortex_leg`, `pressure_jump_coefficient`, `panel_force`.
- [x] Loads: dimensional force/moment names, full coefficient names, and
  explicit reference quantities.
- [x] Diagnostics: full energy, enstrophy, helicity, impulse, stabilization,
  spacing, and convergence names.

## Implementation inventory

### FVM

- [x] Core state, history state, time stepping, diagnostics, logs, and public
  methods use canonical names.
- [x] Assembly, equation solves, pressure correction, turbulence, immersed
  boundary, mesh geometry, partitioning, and coupling interfaces are aligned.
- [x] Serial and partitioned checkpoints use strict current schemas.
- [x] VTK, CSV, profiling, monitoring, and sampler outputs validate names.

### VPM

- [x] Particle state, solvers, physics, acceleration, diffusion,
  stabilization, initial conditions, and diagnostics are aligned.
- [x] Particle vector strength and scalar circulation remain physically
  distinct everywhere.
- [x] HDF5, XDMF, VTP, CSV, JSON, JSONL, monitoring, and checkpoint outputs
  use the registry and current schema version.
- [x] Current VPM checkpoint readers require exact dataset shapes and names.

### VLM and panel solvers

- [x] Geometry, lattice state, influence matrices, right-hand sides,
  kinematics, wake state, loads, and diagnostics use complete names.
- [x] VTK arrays and force/loading records use canonical names and schema
  metadata.
- [x] Compact cross-module matrix, velocity, and coefficient names were
  expanded.

### Coupler

- [x] Boundary geometry, velocity, normal velocity, tangential gradient,
  vorticity transfer, and substep interfaces are canonical.
- [x] Coupled manifests name and hash the FVM, VPM, VPM XDMF, and VPM boundary
  condition artifacts.
- [x] Boundary NPZ payloads use explicit `has_*` flags and schema versions.

### Tutorials, archives, and generated artifacts

- [x] Solver/tutorial directories and filenames are lower snake case.
- [x] Setup, plotting, validation, benchmark, experiment, and shell scripts
  use canonical keys and APIs.
- [x] ParaView state files use canonical physical array identifiers.
- [x] CSV headers, VTK/VTS arrays, PVD references, XDMF attributes, JSON keys,
  NPZ fields, and HDF5 datasets/attributes were converted.
- [x] Both tracked cube-flow run archives were materialized at their final
  canonical paths with their payload names converted.
- [x] Present ignored sample archives and coupled solution checkpoints were
  converted in place to the current schema.
- [x] The one-time conversion utility is absent from the final tree.

### Tests and documentation

- [x] Tests and fixtures call only canonical APIs and assert canonical output
  contracts.
- [x] Dedicated negative assertions verify that removed public aliases do not
  reappear; they provide no runtime compatibility behavior.
- [x] README, installation, contributor, agent, and tutorial documentation use
  canonical paths and terms.

## Final static gate

- [x] Parse every Python file with the standard AST parser.
- [x] Run Ruff lint and formatting checks.
- [x] Run the source/path nomenclature scan.
- [x] Run the generated text/NPZ/HDF5 nomenclature scan.
- [x] Parse JSON and XML-family artifacts.
- [x] Validate registry uniqueness and serialized-name coverage.
- [x] Run Pyrefly on the changed public API surface.
- [x] Run the Git whitespace/error check.
- [x] Confirm the staged tree contains no conversion utility or unintended
  generated checkpoint payloads.
- [x] Commit the fully audited one-way conversion.

No dynamic tests or debugger sessions are part of this project pass, per the
explicit execution constraint.
