# Physical-variable rename manifest

This is the authoritative old-to-canonical mapping used by the one-way rename.
Old spellings are documentation only: they are not accepted by runtime code,
serializers, adapters, or migration readers.

## Physical fields

| Removed spelling | Canonical spelling |
| --- | --- |
| `flow_time`, index-valued `time_step` | `time`, `step` |
| `dt`, `delta_t`, duration-valued `time_step` | `time_step_size` |
| `observed_dt` | `observed_time_step_size` |
| `fvm_substeps` | `n_fvm_substeps` |
| `cfl`, `Co` | `courant_number` |
| `cfl_max`, `max_cfl` | `max_courant_number` |
| `backup_frequency` | `checkpoint_interval_steps` |
| ordinary-output `BACKUP_*` | `OUTPUT_*` |
| restart-state `BACKUP_*` | `CHECKPOINT_*` |
| bare coordinate `x`, `y`, `z` | `position_x`, `position_y`, `position_z` |
| geometric `center`, `centre`, `centroid` | `cell_centre`, `face_centre`, `panel_centre`, or another meaning-specific centre |
| weighted vortex location | `vortex_centroid` |
| `PanelCenter`, `PanelChord`, `BoundLeg` | `panel_centre`, `panel_chord`, `bound_vortex_leg` |
| `Normal`, `Area`, `IsTE`, `IsLE` | `normal`, `area`, `is_trailing_edge`, `is_leading_edge` |
| `U`, `V`, `Velocity` | `velocity` |
| `Ux`, `Uy`, `Uz` | `velocity_x`, `velocity_y`, `velocity_z` |
| `V_inf`, `U_inf` | `freestream_velocity` |
| `U_inf_mag` | `freestream_speed` |
| `grad_u`, `VelocityGradient` | `velocity_gradient` |
| `dudx` … `dwdz` | `velocity_gradient_xx` … `velocity_gradient_zz` |
| `Sxx` … `Szz`, `StrainRate` | `strain_rate_xx` … `strain_rate_zz`, `strain_rate` |
| `VelocityMagnitude`, `BoundVelocity` | `velocity_magnitude`, `bound_vortex_velocity` |
| `compute_target_velocities` | `compute_velocity_at_points` |
| `compute_target_velocity_gradients` | `compute_velocity_gradient_at_points` |
| `compute_complete_target_velocity_and_gradients` | `compute_velocity_and_gradient_at_points` |
| FVM `p`, `Pressure` | `kinematic_pressure` |
| `phi`, `face_flux` | `volumetric_face_flux` |
| `Cp`, `DeltaCp` | `pressure_coefficient`, `pressure_jump_coefficient` |
| kinematic `PressureGradient` | `kinematic_pressure_gradient` |
| `nu`, `viscosity` | `kinematic_viscosity` |
| `nut`, `turbulent_viscosity`, `viscosity_turbulent` | `eddy_viscosity` |
| `viscosity_effective` | `effective_viscosity` |
| `mu` | `dynamic_viscosity` |
| `Cs`, `Cw` | `smagorinsky_coefficient`, `wale_coefficient` |
| `omega`, `Vorticity` | `vorticity` |
| `omega_x`, `omega_y`, `omega_z` | `vorticity_x`, `vorticity_y`, `vorticity_z` |
| VPM `circulation`, `gamma`, `strength` | `vortex_strength` |
| VPM `max_gamma`, `max_strength` | `max_vortex_strength_magnitude` |
| particle `radius` | `core_radius` |
| `VortexStrength`, `CoreRadius` | `vortex_strength`, `core_radius` |
| `GroupID`, `ZoneID` | `group_id`, `zone_id` |
| `vector_circulation` | `net_vortex_strength` |
| `length_strength`, `strength_magnitude` | `vortex_strength_magnitude_sum` |
| `Fx`, `Fy`, `Fz` | `force_x`, `force_y`, `force_z` |
| `Fpx`, `Fpy`, `Fpz` | `pressure_force_x`, `pressure_force_y`, `pressure_force_z` |
| `Fvx`, `Fvy`, `Fvz` | `viscous_force_x`, `viscous_force_y`, `viscous_force_z` |
| `Ftx`, `Fty`, `Ftz` | `total_force_x`, `total_force_y`, `total_force_z` |
| `Mx`, `My`, `Mz` | `moment_x`, `moment_y`, `moment_z` |
| `L`, `D`, `C` | `lift`, `drag`, `side_force` |
| `CL`, `Cd`, `CD`, `CC`, `Cz` | `lift_coefficient`, `drag_coefficient`, `side_force_coefficient` |
| `Cl`, `Cm`, `Cn` | `rolling_moment_coefficient`, `pitching_moment_coefficient`, `yawing_moment_coefficient` |
| `q` | `dynamic_pressure` |
| `S_ref`, `c_ref`, `b_ref`, `r_ref` | `reference_area`, `reference_chord`, `reference_span`, `reference_point` |
| `PanelForce`, `DoubletStrength` | `panel_force`, `doublet_strength` |
| global `kinetic_energy`, `enstrophy`, `helicity` | `total_kinetic_energy`, `total_enstrophy`, `total_helicity` |
| `enstrophy_test` | `test_filtered_enstrophy` |
| `dEdt`, `neg_nu_enstrophy` | `kinetic_energy_rate`, `viscous_kinetic_energy_rate` |
| `impulse_x` | `linear_impulse_x` |
| `angle_rad` | `angle_radians` |
| `strength_misalignment_deg` | `vortex_strength_misalignment_degrees` |
| `stabilization_circulation_error` | `stabilization_vortex_strength_error` |
| `stabilization_strength_growth` | `stabilization_vortex_strength_growth` |

## State, packages, and paths

| Removed spelling/path | Canonical spelling/path |
| --- | --- |
| `face_flux`, `face_flux_old`, `face_flux_older` | `volumetric_face_flux`, `volumetric_face_flux_old`, `volumetric_face_flux_older` |
| `vpm_bc` | `vpm_boundary_condition_state` |
| `_velocity_bc_prev` | `_previous_boundary_velocity` |
| `source/solvers/FVM`, `source/solvers/VPM` | `source/solvers/fvm`, `source/solvers/vpm` |
| `lambOseenVortex`, `vortexRing`, `vortexInteractions` | `lamb_oseen_vortex`, `vortex_ring`, `vortex_interactions` |
| `quadCopter`, `rotorFlow`, `deltaWing`, `flatPlate` | `quadcopter`, `rotor_flow`, `delta_wing`, `flat_plate` |
| camel-case tutorial setup filenames | lower `snake_case` setup filenames |

## Current stored formats

| Artifact | Current contract |
| --- | --- |
| FVM serial NPZ | format 7 with exact canonical state and metadata keys |
| FVM partitioned NPZ | format 5 with serial-equivalent physical keys |
| VPM HDF5/XDMF | checkpoint format 7.0 with identical canonical field names |
| Coupled checkpoint | format 9 with named, hashed child artifacts |
| VTK-family output | canonical arrays owned by each exporter |
| CSV/JSON/JSONL output | canonical headers and keys |

The canonical package and tutorial path trees are complete. The tracked cube
run archives live under `tutorials/coupled_fvm_vpm/cube_flow/run_archives/`,
and restart-qualified coupled state lives under `solution/checkpoints/`.
