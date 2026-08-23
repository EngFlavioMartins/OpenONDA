# OpenONDA canonical physical nomenclature

This is the normative naming contract for every solver, coupler, tutorial,
archive, checkpoint, generated artifact, and post-processing tool. The
machine-readable authority is `source/schemas/physical_fields.py`.

## Naming grammar

- Use lower `snake_case` for physical names in Python and every serialized
  schema.
- Use British `centre` consistently for geometry, including `cell_centre`,
  `face_centre`, `panel_centre`, `rotation_centre`, and `centreline`.
- Use singular field names and `n_*` for counts. Use `is_*` and `has_*` for
  booleans.
- Use `_old` and `_older` only for time-history states.
- Use `_x`, `_y`, and `_z` for vector components. Tensor components use
  row-major `_xx`, `_xy`, `_xz`, `_yx`, `_yy`, `_yz`, `_zx`, `_zy`, `_zz`.
- A name always preserves the same quantity, units, rank, component order,
  and physical location.
- Compact mathematical symbols are confined to isolated numerical kernels.
  They never cross a function API, module boundary, log record, or file
  boundary.

## Canonical quantities

| Domain | Canonical identifiers | Meaning |
| --- | --- | --- |
| Time | `time`, `nondimensional_time`, `step`, `time_step_size`, `accepted_time_step_size`, `observed_time_step_size` | physical time and explicit step-size semantics |
| Counts | `n_particles_total`, `n_particles_shed`, `n_particles_removed`, `n_fvm_substeps`, `n_global_cells`, `n_ranks` | populations and partition sizes |
| Geometry | `position`, `vertex_position`, `cell_centre`, `face_centre`, `panel_centre`, `geometry_centre`, `rotation_centre`, `vortex_centroid` | Cartesian locations in metres |
| Mesh measures | `face_area`, `face_area_vector`, `cell_volume`, `particle_volume`, `area`, `normal` | oriented and scalar geometric measures |
| Velocity | `velocity`, `freestream_velocity`, `background_velocity`, `prescribed_velocity`, `kinematic_velocity`, `bound_vortex_velocity`, `angular_velocity` | velocity fields with explicit physical roles |
| Derivatives | `velocity_gradient`, `strain_rate`, `vorticity`, `kinematic_pressure_gradient` | tensors and curls in row-major component order |
| Pressure and flux | `kinematic_pressure`, `pressure`, `kinematic_pressure_gradient`, `pressure_gradient`, `volumetric_face_flux`, `mass_flux`, `courant_number`, `max_courant_number` | pressure, transport, and stability quantities |
| Viscosity | `kinematic_viscosity`, `dynamic_viscosity`, `eddy_viscosity`, `effective_viscosity` | molecular, turbulent, and combined viscosity |
| Vortex particles | `vortex_strength`, `vortex_strength_magnitude`, `core_radius`, `particle_volume`, `group_id`, `zone_id` | VPM particle state |
| Line vortices | `circulation`, `bound_circulation`, `wake_circulation`, `tube_circulation`, `doublet_strength` | scalar line or panel strengths |
| Loads | `force_x`, `force_y`, `force_z`, `moment_x`, `moment_y`, `moment_z`, `lift`, `drag`, `side_force` | dimensional loads |
| Coefficients | `lift_coefficient`, `drag_coefficient`, `side_force_coefficient`, `rolling_moment_coefficient`, `pitching_moment_coefficient`, `yawing_moment_coefficient` | nondimensional loads |
| References | `reference_velocity`, `reference_area`, `reference_length`, `reference_chord`, `reference_span`, `reference_point` | load normalization metadata |
| Integrals | `total_kinetic_energy`, `total_enstrophy`, `total_helicity`, `kinetic_energy_rate`, `viscous_kinetic_energy_rate` | global diagnostics |

The registry contains the complete set, including component fields,
stabilization diagnostics, VLM/panel arrays, and visualization fields.

## Semantic distinctions

- `kinematic_pressure` is pressure divided by density; `pressure` is measured
  in pascals.
- `kinematic_pressure_gradient` is acceleration in m/s²;
  `pressure_gradient` and its convective, temporal, and viscous contributions
  are dimensional gradients in Pa/m.
- `volumetric_face_flux` is velocity dotted with `face_area_vector` and is not
  `mass_flux`.
- `vortex_strength` is the VPM particle vector in m³/s. `circulation` is a
  scalar line or bound-vortex quantity in m²/s.
- `time_step_size` is the configured or next step,
  `accepted_time_step_size` is the last committed advance, and
  `observed_time_step_size` is a measured output interval.
- `velocity_gradient` uses component order `(xx, xy, xz, yx, yy, yz, zx, zy,
  zz)` whenever flattened.

## Schema policy

Every writer validates physical names against the registry and stamps the
output with `physical_field_schema_version`. Readers accept exactly the current
format and physical-field schema. There are no aliases, alternate keys,
fallback spellings, migration readers, or dual-format writers.

VPM checkpoints store particle volume only as
`particles/particle_volume`; the generic dataset name `particles/volume` is
not part of the current format.
