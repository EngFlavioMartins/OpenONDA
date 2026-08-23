"""Machine-readable canonical physical-field nomenclature.

The registry is intentionally dependency-free so FVM, VPM, VLM, the coupler,
and offline tools can import it without importing a numerical backend. Writers
should validate keys before serializing them and include :data:`SCHEMA_VERSION`
in their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final[str] = "physical-fields-1"


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    """Definition of one canonical field name."""

    name: str
    description: str
    units: str
    rank: int
    location: str
    components: tuple[str, ...] = ()


def _field(
    name: str,
    description: str,
    units: str,
    rank: int,
    location: str,
    components: tuple[str, ...] = (),
) -> FieldDefinition:
    return FieldDefinition(name, description, units, rank, location, components)


def _build_registry(
    definitions: tuple[FieldDefinition, ...],
) -> dict[str, FieldDefinition]:
    registry: dict[str, FieldDefinition] = {}
    for definition in definitions:
        if definition.name in registry:
            raise RuntimeError(f"Duplicate physical-field definition {definition.name!r}")
        registry[definition.name] = definition
    return registry


FIELD_REGISTRY: Final[dict[str, FieldDefinition]] = _build_registry(
    (
        _field("time", "physical simulation time", "s", 0, "global"),
        _field("nondimensional_time", "nondimensional physical time", "1", 0, "global"),
        _field("integration_time", "particle or streamline integration time", "s", 0, "point"),
        _field("step", "committed time-step index", "1", 0, "global"),
        _field("time_step_size", "configured or next time-step size", "s", 0, "global"),
        _field("accepted_time_step_size", "last committed time-step size", "s", 0, "global"),
        _field("observed_time_step_size", "measured output-frame interval", "s", 0, "global"),
        _field("n_fvm_substeps", "number of FVM substeps", "1", 0, "global"),
        _field("n_particles_total", "total number of VPM particles", "1", 0, "global"),
        _field("n_particles_shed", "particles shed during a step", "1", 0, "global"),
        _field("n_particles_removed", "particles removed during a step", "1", 0, "global"),
        _field("n_global_cells", "total number of cells across all partitions", "1", 0, "global"),
        _field("n_ranks", "number of parallel solver ranks", "1", 0, "global"),
        _field("global_cell_id", "global FVM cell identifier", "1", 0, "cell"),
        _field("global_face_id", "global FVM face identifier", "1", 0, "face"),
        _field("global_point_id", "global mesh-point identifier", "1", 0, "point"),
        _field("position", "Cartesian position", "m", 1, "point", ("x", "y", "z")),
        _field("vertex_position", "surface vertex position", "m", 1, "vertex", ("x", "y", "z")),
        _field("cell_centre", "FVM cell centre", "m", 1, "cell", ("x", "y", "z")),
        _field("face_centre", "FVM face centre", "m", 1, "face", ("x", "y", "z")),
        _field("panel_centre", "panel centre", "m", 1, "panel", ("x", "y", "z")),
        _field("geometry_centre", "reference geometry centre", "m", 1, "surface", ("x", "y", "z")),
        _field("rotation_centre", "rigid-body rotation centre", "m", 1, "surface", ("x", "y", "z")),
        _field("vortex_centroid", "weighted vortex centroid", "m", 1, "vortex", ("x", "y", "z")),
        _field("position_magnitude", "Cartesian position norm", "m", 0, "point"),
        _field("velocity", "fluid velocity", "m/s", 1, "field", ("x", "y", "z")),
        _field("freestream_velocity", "freestream velocity", "m/s", 1, "global", ("x", "y", "z")),
        _field(
            "background_velocity",
            "resolved background-flow velocity",
            "m/s",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "kinematic_velocity",
            "surface velocity imposed by rigid-body kinematics",
            "m/s",
            1,
            "surface",
            ("x", "y", "z"),
        ),
        _field(
            "angular_velocity",
            "rigid-body angular velocity",
            "rad/s",
            1,
            "surface",
            ("x", "y", "z"),
        ),
        _field(
            "prescribed_velocity",
            "prescribed boundary or body velocity",
            "m/s",
            1,
            "boundary",
            ("x", "y", "z"),
        ),
        _field("freestream_speed", "freestream speed", "m/s", 0, "global"),
        _field("velocity_magnitude", "velocity Euclidean norm", "m/s", 0, "field"),
        _field(
            "nondimensional_velocity",
            "velocity normalized by a declared reference velocity",
            "1",
            0,
            "field",
        ),
        _field("plane_velocity", "velocity sampled on a plane", "m/s", 1, "field", ("x", "y", "z")),
        _field("max_velocity_magnitude", "maximum velocity magnitude", "m/s", 0, "global"),
        _field("mean_velocity_magnitude", "mean velocity magnitude", "m/s", 0, "global"),
        _field("vorticity", "curl of velocity", "1/s", 1, "field", ("x", "y", "z")),
        _field("vorticity_magnitude", "vorticity Euclidean norm", "1/s", 0, "field"),
        _field("velocity_gradient", "velocity gradient", "1/s", 2, "field"),
        _field("strain_rate", "symmetric velocity gradient", "1/s", 2, "field"),
        _field("kinematic_pressure", "pressure divided by density", "m^2/s^2", 0, "cell"),
        _field("pressure", "thermodynamic pressure", "Pa", 0, "field"),
        _field(
            "pressure_gradient",
            "gradient of thermodynamic pressure",
            "Pa/m",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "convective_pressure_gradient",
            "convective contribution to the thermodynamic-pressure gradient",
            "Pa/m",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "temporal_pressure_gradient",
            "temporal contribution to the thermodynamic-pressure gradient",
            "Pa/m",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "viscous_pressure_gradient",
            "viscous contribution to the thermodynamic-pressure gradient",
            "Pa/m",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "kinematic_pressure_gradient",
            "gradient of kinematic pressure",
            "m/s^2",
            1,
            "cell",
            ("x", "y", "z"),
        ),
        _field(
            "kinematic_pressure_gradient_magnitude",
            "kinematic-pressure-gradient Euclidean norm",
            "m/s^2",
            0,
            "field",
        ),
        _field(
            "convective_kinematic_pressure_gradient",
            "convective contribution to the kinematic-pressure gradient",
            "m/s^2",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "temporal_kinematic_pressure_gradient",
            "temporal contribution to the kinematic-pressure gradient",
            "m/s^2",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field(
            "viscous_kinematic_pressure_gradient",
            "viscous contribution to the kinematic-pressure gradient",
            "m/s^2",
            1,
            "field",
            ("x", "y", "z"),
        ),
        _field("volumetric_face_flux", "velocity dot face area vector", "m^3/s", 0, "face"),
        _field("mass_flux", "mass flow through a face", "kg/s", 0, "face"),
        _field("pressure_coefficient", "normalised pressure", "1", 0, "field"),
        _field("min_pressure_coefficient", "minimum pressure coefficient", "1", 0, "global"),
        _field("max_pressure_coefficient", "maximum pressure coefficient", "1", 0, "global"),
        _field("pressure_jump_coefficient", "panel pressure jump coefficient", "1", 0, "panel"),
        _field(
            "max_leading_edge_suction_parameter",
            "maximum leading-edge suction parameter",
            "1",
            0,
            "global",
        ),
        _field("courant_number", "local Courant number", "1", 0, "cell"),
        _field("max_courant_number", "maximum Courant number", "1", 0, "global"),
        _field("max_continuity_error", "maximum absolute velocity divergence", "1/s", 0, "global"),
        _field(
            "sum_absolute_continuity_error",
            "sum of absolute cell volumetric-flux residuals",
            "m^3/s",
            0,
            "global",
        ),
        _field(
            "net_boundary_volumetric_flux", "net boundary volumetric flux", "m^3/s", 0, "global"
        ),
        _field("kinematic_viscosity", "molecular kinematic viscosity", "m^2/s", 0, "field"),
        _field(
            "max_kinematic_viscosity", "maximum molecular kinematic viscosity", "m^2/s", 0, "global"
        ),
        _field("dynamic_viscosity", "dynamic viscosity", "Pa*s", 0, "field"),
        _field("eddy_viscosity", "turbulent kinematic viscosity", "m^2/s", 0, "field"),
        _field(
            "effective_viscosity", "molecular plus eddy kinematic viscosity", "m^2/s", 0, "field"
        ),
        _field("mean_effective_viscosity", "mean effective viscosity", "m^2/s", 0, "global"),
        _field("max_effective_viscosity", "maximum effective viscosity", "m^2/s", 0, "global"),
        _field("mean_eddy_viscosity", "mean eddy viscosity", "m^2/s", 0, "global"),
        _field("max_eddy_viscosity", "maximum eddy viscosity", "m^2/s", 0, "global"),
        _field("smagorinsky_coefficient", "Smagorinsky coefficient", "1", 0, "field"),
        _field("wale_coefficient", "WALE coefficient", "1", 0, "field"),
        _field("sigma_coefficient", "Sigma-model coefficient", "1", 0, "field"),
        _field(
            "subgrid_kinetic_energy_coefficient",
            "subgrid kinetic-energy coefficient",
            "1",
            0,
            "field",
        ),
        _field(
            "subgrid_dissipation_coefficient", "subgrid dissipation coefficient", "1", 0, "field"
        ),
        _field(
            "test_filter_width_ratio_squared", "squared test-filter width ratio", "1", 0, "field"
        ),
        _field(
            "vortex_strength",
            "VPM particle vector strength",
            "m^3/s",
            1,
            "particle",
            ("x", "y", "z"),
        ),
        _field(
            "bound_vortex_strength",
            "bound-vortex vector strength",
            "m^3/s",
            1,
            "panel",
            ("x", "y", "z"),
        ),
        _field(
            "wake_vortex_strength",
            "wake-vortex vector strength",
            "m^3/s",
            1,
            "particle",
            ("x", "y", "z"),
        ),
        _field(
            "filament_reference_vortex_strength",
            "filament reference vortex strength",
            "m^3/s",
            1,
            "particle",
            ("x", "y", "z"),
        ),
        _field("filament_reference_length", "filament reference length", "m", 0, "particle"),
        _field("vortex_strength_magnitude", "particle strength norm", "m^3/s", 0, "particle"),
        _field(
            "vortex_strength_magnitude_sum", "sum of particle strength norms", "m^3/s", 0, "global"
        ),
        _field(
            "net_vortex_strength",
            "vector sum of particle strengths",
            "m^3/s",
            1,
            "global",
            ("x", "y", "z"),
        ),
        _field(
            "net_vortex_strength_magnitude", "norm of net vortex strength", "m^3/s", 0, "global"
        ),
        _field("core_radius", "vortex kernel radius", "m", 0, "particle"),
        _field("normal", "outward unit normal", "1", 1, "surface", ("x", "y", "z")),
        _field("area", "face or panel area", "m^2", 0, "surface"),
        _field("face_area", "FVM face area", "m^2", 0, "face"),
        _field(
            "face_area_vector", "oriented FVM face area vector", "m^2", 1, "face", ("x", "y", "z")
        ),
        _field("cell_volume", "FVM cell volume", "m^3", 0, "cell"),
        _field("particle_volume", "VPM particle volume", "m^3", 0, "particle"),
        _field("panel_chord", "panel chord vector", "m", 1, "panel", ("x", "y", "z")),
        _field("bound_vortex_leg", "bound vortex leg vector", "m", 1, "panel", ("x", "y", "z")),
        _field("is_trailing_edge", "trailing-edge flag", "1", 0, "panel"),
        _field("is_leading_edge", "leading-edge flag", "1", 0, "panel"),
        _field("group_id", "particle group identifier", "1", 0, "particle"),
        _field("zone_id", "particle zone identifier", "1", 0, "particle"),
        _field("circulation", "scalar line or bound-vortex circulation", "m^2/s", 0, "vortex"),
        _field("tube_circulation", "vortex-tube circulation", "m^2/s", 0, "vortex"),
        _field("bound_circulation", "bound circulation", "m^2/s", 0, "panel"),
        _field("wake_circulation", "wake circulation", "m^2/s", 0, "vortex"),
        _field("doublet_strength", "panel doublet strength", "m^3/s", 0, "panel"),
        _field(
            "bound_vortex_velocity",
            "velocity induced by bound vortices",
            "m/s",
            1,
            "panel",
            ("x", "y", "z"),
        ),
        _field("panel_force", "force carried by a panel", "N", 1, "panel", ("x", "y", "z")),
        _field("force_x", "force x component", "N", 0, "body"),
        _field("force_y", "force y component", "N", 0, "body"),
        _field("force_z", "force z component", "N", 0, "body"),
        _field("pressure_force_x", "pressure force x component", "N", 0, "body"),
        _field("pressure_force_y", "pressure force y component", "N", 0, "body"),
        _field("pressure_force_z", "pressure force z component", "N", 0, "body"),
        _field("viscous_force_x", "viscous force x component", "N", 0, "body"),
        _field("viscous_force_y", "viscous force y component", "N", 0, "body"),
        _field("viscous_force_z", "viscous force z component", "N", 0, "body"),
        _field("total_force_x", "total force x component", "N", 0, "body"),
        _field("total_force_y", "total force y component", "N", 0, "body"),
        _field("total_force_z", "total force z component", "N", 0, "body"),
        _field("moment_x", "moment x component", "N*m", 0, "body"),
        _field("moment_y", "moment y component", "N*m", 0, "body"),
        _field("moment_z", "moment z component", "N*m", 0, "body"),
        _field("lift", "lift force", "N", 0, "body"),
        _field("drag", "drag force", "N", 0, "body"),
        _field("side_force", "side force", "N", 0, "body"),
        _field("thrust", "thrust force", "N", 0, "body"),
        _field("torque", "torque moment", "N*m", 0, "body"),
        _field("power", "mechanical power", "W", 0, "body"),
        _field("lift_coefficient", "lift coefficient", "1", 0, "body"),
        _field("drag_coefficient", "drag coefficient", "1", 0, "body"),
        _field("side_force_coefficient", "side-force coefficient", "1", 0, "body"),
        _field("force_coefficient_x", "force coefficient x component", "1", 0, "body"),
        _field("force_coefficient_y", "force coefficient y component", "1", 0, "body"),
        _field("force_coefficient_z", "force coefficient z component", "1", 0, "body"),
        _field("rolling_moment_coefficient", "rolling-moment coefficient", "1", 0, "body"),
        _field("pitching_moment_coefficient", "pitching-moment coefficient", "1", 0, "body"),
        _field("yawing_moment_coefficient", "yawing-moment coefficient", "1", 0, "body"),
        _field(
            "rolling_moment_coefficient_quarter_chord",
            "rolling-moment coefficient at quarter chord",
            "1",
            0,
            "body",
        ),
        _field(
            "pitching_moment_coefficient_quarter_chord",
            "pitching-moment coefficient at quarter chord",
            "1",
            0,
            "body",
        ),
        _field(
            "yawing_moment_coefficient_quarter_chord",
            "yawing-moment coefficient at quarter chord",
            "1",
            0,
            "body",
        ),
        _field("dynamic_pressure", "freestream dynamic pressure", "Pa", 0, "global"),
        _field("reference_velocity", "force reference velocity", "m/s", 0, "global"),
        _field("reference_area", "force reference area", "m^2", 0, "global"),
        _field("reference_length", "force reference length", "m", 0, "global"),
        _field("reference_chord", "force reference chord", "m", 0, "global"),
        _field("reference_span", "force reference span", "m", 0, "global"),
        _field("reference_point", "moment reference point", "m", 1, "global", ("x", "y", "z")),
        _field("min_particle_spacing", "minimum particle spacing", "m", 0, "global"),
        _field("max_particle_spacing", "maximum particle spacing", "m", 0, "global"),
        _field("mean_particle_spacing", "mean particle spacing", "m", 0, "global"),
        _field(
            "particle_spacing_ratio", "minimum to maximum particle spacing ratio", "1", 0, "global"
        ),
        _field("mean_core_radius", "mean particle core radius", "m", 0, "global"),
        _field(
            "mean_overlap_ratio", "mean particle spacing to core-radius ratio", "1", 0, "global"
        ),
        _field("max_overlap_ratio", "maximum spacing to core-radius ratio", "1", 0, "global"),
        _field(
            "vorticity_divergence_error", "discrete vorticity divergence error", "1", 0, "global"
        ),
        _field(
            "invariant_projection_correction_ratio",
            "invariant projection correction ratio",
            "1",
            0,
            "global",
        ),
        _field(
            "max_velocity_gradient_magnitude",
            "maximum velocity-gradient magnitude",
            "1/s",
            0,
            "global",
        ),
        _field("slip_error", "velocity boundary slip error", "m/s", 0, "global"),
        _field("total_kinetic_energy", "domain kinetic energy", "J", 0, "global"),
        _field("total_enstrophy", "domain enstrophy", "solver-defined", 0, "global"),
        _field("test_filtered_enstrophy", "test-filtered enstrophy", "solver-defined", 0, "global"),
        _field("total_helicity", "domain helicity", "solver-defined", 0, "global"),
        _field("kinetic_energy_rate", "signed kinetic-energy rate", "J/s", 0, "global"),
        _field("viscous_kinetic_energy_rate", "signed viscous energy rate", "J/s", 0, "global"),
        _field("angle_radians", "angle in radians", "rad", 0, "global"),
        _field(
            "vortex_strength_misalignment_degrees",
            "vortex-strength misalignment",
            "deg",
            0,
            "global",
        ),
        _field(
            "stabilization_vortex_strength_error",
            "stabilization strength error",
            "solver-defined",
            0,
            "global",
        ),
        _field(
            "stabilization_vortex_strength_growth",
            "stabilization strength growth",
            "solver-defined",
            0,
            "global",
        ),
        _field(
            "stabilization_vorticity_growth",
            "latest stabilization vorticity growth",
            "1",
            0,
            "global",
        ),
        _field(
            "max_stabilization_vorticity_growth",
            "maximum stabilization vorticity growth",
            "1",
            0,
            "global",
        ),
        _field("n_stabilization_events", "number of stabilization events", "1", 0, "global"),
        _field("last_stabilization_mechanism", "latest stabilization mechanism", "1", 0, "global"),
        _field(
            "mean_stabilization_kinematic_viscosity",
            "mean stabilization kinematic viscosity",
            "m^2/s",
            0,
            "global",
        ),
        _field(
            "max_stabilization_kinematic_viscosity",
            "maximum stabilization kinematic viscosity",
            "m^2/s",
            0,
            "global",
        ),
        _field(
            "stabilization_kinematic_viscosity_active_fraction",
            "fraction with active stabilization viscosity",
            "1",
            0,
            "global",
        ),
        _field(
            "vortex_strength_closure_error_percent",
            "bound/wake vector-strength closure error",
            "%",
            0,
            "global",
        ),
        _field(
            "bound_vortex_strength_magnitude",
            "bound vector-strength magnitude",
            "m^3/s",
            0,
            "global",
        ),
        _field(
            "wake_vortex_strength_magnitude", "wake vector-strength magnitude", "m^3/s", 0, "global"
        ),
        _field("kutta_joukowski_force_x", "Kutta-Joukowski force x component", "N", 0, "body"),
        _field("kutta_joukowski_force_y", "Kutta-Joukowski force y component", "N", 0, "body"),
        _field("kutta_joukowski_force_z", "Kutta-Joukowski force z component", "N", 0, "body"),
        _field(
            "max_vortex_strength_magnitude", "maximum particle strength norm", "m^3/s", 0, "global"
        ),
        _field("angular_impulse_x", "angular impulse x component", "solver-defined", 0, "global"),
        _field("angular_impulse_y", "angular impulse y component", "solver-defined", 0, "global"),
        _field("angular_impulse_z", "angular impulse z component", "solver-defined", 0, "global"),
        _field("linear_impulse_x", "linear impulse x component", "solver-defined", 0, "global"),
        _field("linear_impulse_y", "linear impulse y component", "solver-defined", 0, "global"),
        _field("linear_impulse_z", "linear impulse z component", "solver-defined", 0, "global"),
        _field(
            "linear_impulse_magnitude", "linear impulse magnitude", "solver-defined", 0, "global"
        ),
        _field("major_radius", "ring major radius", "m", 0, "vortex"),
        _field("impulse_radius", "equivalent ring impulse radius", "m", 0, "vortex"),
    )
)


def canonical_component_names(field_name: str) -> tuple[str, ...]:
    """Return canonical component keys for a vector or tensor field."""
    definition = validate_field_name(field_name)
    if definition.components == ("x", "y", "z"):
        return tuple(f"{field_name}_{axis}" for axis in definition.components)
    if definition.rank == 2:
        return tuple(f"{field_name}_{row}{column}" for row in "xyz" for column in "xyz")
    return ()


def validate_field_name(field_name: str) -> FieldDefinition:
    """Return a field definition or raise a clear schema error."""
    try:
        return FIELD_REGISTRY[field_name]
    except KeyError as error:
        for base_name, definition in FIELD_REGISTRY.items():
            if definition.components == ("x", "y", "z"):
                if field_name in {f"{base_name}_{axis}" for axis in "xyz"}:
                    return FieldDefinition(
                        field_name,
                        f"{definition.description} component",
                        definition.units,
                        0,
                        definition.location,
                    )
            elif definition.rank == 2 and field_name in {
                f"{base_name}_{row}{column}" for row in "xyz" for column in "xyz"
            }:
                return FieldDefinition(
                    field_name,
                    f"{definition.description} component",
                    definition.units,
                    0,
                    definition.location,
                )
        raise ValueError(
            f"Unknown physical field {field_name!r}; add it to the canonical registry"
        ) from error


def validate_field_contract(
    field_name: str,
    *,
    units: str,
    rank: int,
    location: str,
    components: tuple[str, ...] | None = None,
) -> FieldDefinition:
    """Validate all declared metadata for a canonical physical field."""
    definition = validate_field_name(field_name)
    mismatches: list[str] = []
    if units != definition.units:
        mismatches.append(f"units {units!r} != {definition.units!r}")
    if rank != definition.rank:
        mismatches.append(f"rank {rank!r} != {definition.rank!r}")
    if location != definition.location:
        mismatches.append(f"location {location!r} != {definition.location!r}")
    if components is not None and components != definition.components:
        mismatches.append(f"components {components!r} != canonical order {definition.components!r}")
    if mismatches:
        raise ValueError(f"Invalid contract for {field_name!r}: " + "; ".join(mismatches))
    return definition


_NONPHYSICAL_OUTPUT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "vtkGhostType",
        "vtkOriginalCellIds",
        "vtkOriginalPointIds",
        "vtkValidPointMask",
        "surface_ordering",
        "physical_field_schema_version",
    }
)


def validate_serialized_field_name(field_name: str) -> str:
    """Validate a field name crossing a serializer boundary.

    Every physical field must be present in the canonical registry.
    Non-physical VTK bookkeeping names are allowed only for the adapters that
    require them.
    """
    if field_name in _NONPHYSICAL_OUTPUT_NAMES:
        return field_name
    validate_field_name(field_name)
    return field_name
