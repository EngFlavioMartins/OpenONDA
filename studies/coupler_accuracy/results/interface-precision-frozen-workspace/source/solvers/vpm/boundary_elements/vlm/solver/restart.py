"""Exact VLM continuation state inside the owning VPM numerical backup."""

import hashlib
import inspect
import json
import marshal
from types import CodeType

import numpy as np

from ....io.manifest import _manifest_value

_FIELDS = (
    "panel_corner_position",
    "vortex_point_position",
    "collocation_point",
    "bound_vortex_midpoint",
    "normal",
    "area",
    "circulation",
    "circulation_old",
    "cumulative_circulation",
    "cumulative_circulation_old",
    "velocity",
    "bound_vortex_velocity",
    "kinematic_velocity",
    "external_velocity",
    "bound_kinematic_velocity",
    "bound_external_velocity",
    "relative_velocity",
    "bound_relative_velocity",
    "panel_force",
    "unsteady_panel_force",
    "unsteady_pressure_jump_coefficient",
    "panel_moment_correction",
    "pressure_coefficient",
    "leading_edge_suction_parameter",
)
_MOTION_FIELDS = ("current_position", "current_orientation", "rotation_centre")
_OUTPUT_IDENTITY_FIELDS = ("logging_interval_steps", "sample_surface_forces")


def _portable_code(code):
    """Retain executable callback identity while ignoring its source location."""
    constants = tuple(
        _portable_code(value) if isinstance(value, CodeType) else value for value in code.co_consts
    )
    return code.replace(co_filename="<motion>", co_firstlineno=0, co_consts=constants)


def _identity_controls(vlm, *, output_controls=None, include_output_controls=True):
    """Return the deterministic VLM controls used by restart identities."""
    controls = _manifest_value(vlm.setup)
    # Sampling does not change evolution, including for legacy checkpoints.
    controls.pop("surface_diagnostics_interval_steps", None)
    # The setup describes user choices; this explicit runtime contract also
    # fingerprints the geometric source/trace/transport operators so a
    # checkpoint cannot silently continue under a different field model.
    field_contract = getattr(vlm, "field_contract", None)
    if field_contract is not None and hasattr(field_contract, "as_dict"):
        controls["field_contract"] = field_contract.as_dict()
    # Omission retains the previous shedding rule and restart identity.
    if controls["wake_core_overlap"] is None:
        controls.pop("wake_core_overlap")
    for surface in controls["surfaces"]:
        # Geometry and reference values are hashed below; a moved case retains
        # the same numerical identity without depending on an absolute filename.
        surface.pop("surface")
        if not include_output_controls:
            surface.pop("sample_forces", None)
    if not include_output_controls:
        for name in _OUTPUT_IDENTITY_FIELDS:
            controls.pop(name, None)
    elif output_controls is not None:
        controls["logging_interval_steps"] = output_controls["logging_interval_steps"]
        controls["sample_surface_forces"] = output_controls["sample_surface_forces"]
        for surface, sample_forces in zip(
            controls["surfaces"],
            output_controls.get("surface_sample_forces", ()),
            strict=True,
        ):
            surface["sample_forces"] = sample_forces
    return controls


def _identity_digest(vlm, controls):
    """Hash controls, callbacks and generated geometry for one identity variant."""
    controls["geometry_references"] = getattr(
        vlm,
        "_restart_geometry_references",
        _manifest_value(vlm.aircraft.refs),
    )
    digest = hashlib.sha256(json.dumps(controls, sort_keys=True, allow_nan=False).encode())
    # Generic motion callbacks can depend on closed-over phase and case globals.
    # Record their code and captured construction values without executing them.
    for surface in vlm.setup.surfaces:
        motion = surface.kinematics
        for name in ("velocity_function", "angular_velocity_function"):
            function = getattr(motion, name, None)
            if inspect.isfunction(function):
                closure = inspect.getclosurevars(function)
                digest.update(marshal.dumps(_portable_code(function.__code__)))
                captured = _manifest_value({**closure.globals, **closure.nonlocals})
                digest.update(json.dumps(captured, sort_keys=True, allow_nan=False).encode())
    for name in ("panel_corner_position", "normal"):
        digest.update(getattr(vlm.lattice, name).to_numpy()[: vlm.lattice.n_panels].tobytes())
    return digest.hexdigest()


def restart_identity(vlm, *, output_controls=None):
    """Hash declared physics and output controls plus generated initial geometry."""
    return _identity_digest(vlm, _identity_controls(vlm, output_controls=output_controls))


def restart_physics_identity(vlm):
    """Hash only VLM controls that can change solved or emitted physics.

    The coupled VPM owner now requires every accepted step for scientific
    output.  Legacy standalone logging and per-surface sampling controls are
    therefore output-only compatibility fields and are deliberately omitted
    from this persisted physics identity.
    """
    return _identity_digest(vlm, _identity_controls(vlm, include_output_controls=False))


def write_vlm_restart(vlm, group):
    """Store lattice fields and mutable rigid-motion state without reducing precision."""
    group.attrs["version"] = 7
    group.attrs["identity"] = vlm._restart_identity
    group.attrs["physics_identity"] = getattr(
        vlm,
        "_restart_physics_identity",
        restart_physics_identity(vlm),
    )
    group.attrs["solved"] = vlm._solved
    group.attrs["coupled_mode"] = vlm._coupled_mode
    group.attrs["time"] = vlm._current_time if vlm._current_time is not None else 0.0
    group.attrs["reference_speed"] = vlm.lattice.reference_speed
    group.attrs["reference_velocity"] = getattr(vlm, "_last_reference_velocity", np.zeros(3))
    force_density = getattr(vlm.lattice, "force_density", None)
    if force_density is None:
        force_density = getattr(vlm, "_force_density", None)
    if force_density is not None and np.isfinite(force_density) and force_density > 0.0:
        group.attrs["force_density"] = float(force_density)
    for name in _FIELDS:
        group.create_dataset(
            name, data=getattr(vlm.lattice, name).to_numpy()[: vlm.lattice.n_panels]
        )
    motion = group.create_group("motion")
    for index, (_, kinematics) in enumerate(vlm.surfaces.values()):
        surface = motion.create_group(str(index))
        for name in _MOTION_FIELDS:
            if hasattr(kinematics, name):
                surface.create_dataset(name, data=getattr(kinematics, name))


def validate_vlm_restart(vlm, group, *, legacy_output_controls=None):
    """Validate VLM state before mutation, with explicit legacy output migration."""
    if group is None:
        raise ValueError("VPM backup is missing VLM continuation state")
    version = int(group.attrs.get("version", -1))
    expected_attributes = {
        "version",
        "identity",
        "solved",
        "coupled_mode",
        "time",
        "reference_speed",
        "reference_velocity",
    }
    if version >= 7:
        expected_attributes = expected_attributes | {"physics_identity"}
    allowed_attributes = (expected_attributes, expected_attributes | {"force_density"})
    if set(group.attrs) not in allowed_attributes:
        raise ValueError("VLM restart attributes are incomplete or unknown")
    for name in ("solved", "coupled_mode"):
        if group.attrs[name] not in (False, True):
            raise ValueError(f"Invalid VLM restart flag {name}")
    if np.shape(group.attrs["reference_velocity"]) != (3,):
        raise ValueError("Invalid VLM restart reference velocity")
    if version not in (6, 7):
        raise ValueError(
            "VLM restart uses an incompatible wake/force-state version; "
            "start a new run with this solver"
        )
    identity_migration = None
    expected_physics_identity = getattr(vlm, "_restart_physics_identity", None)
    if expected_physics_identity is None:
        expected_physics_identity = restart_physics_identity(vlm)
    if version >= 7 and group.attrs.get("physics_identity") != expected_physics_identity:
        raise ValueError("VLM restart physics configuration does not match this solver")
    if group.attrs.get("identity") != vlm._restart_identity:
        if version != 6 or legacy_output_controls is None:
            raise ValueError("VLM restart geometry or configuration does not match this solver")
        candidate = restart_identity(vlm, output_controls=legacy_output_controls)
        migration_kind = "output_only"
        if candidate != group.attrs.get("identity"):
            # Version 6 predates explicit field/observer declarations. Only
            # the unchanged default lagged operator can reproduce that schema;
            # all geometry, motion and original physics controls remain hashed.
            from ....config.constants import VLM_EPSILON
            from .field import BoundSurfaceFieldContract

            legacy_contract = BoundSurfaceFieldContract(
                numerical_epsilon=float(VLM_EPSILON), bound_source_radius=float(VLM_EPSILON)
            )
            if vlm.boundary_response != "lagged" or vlm.field_contract != legacy_contract:
                raise ValueError("VLM restart geometry or configuration does not match this solver")
            controls = _identity_controls(vlm, output_controls=legacy_output_controls)
            for name in ("field_contract", "boundary_response", "surface_event_policy"):
                controls.pop(name, None)
            if _identity_digest(vlm, controls) != group.attrs.get("identity"):
                raise ValueError("VLM restart geometry or configuration does not match this solver")
            migration_kind = "legacy_lagged_field_declaration"
        identity_migration = {
            "kind": migration_kind,
            "source_identity": str(group.attrs["identity"]),
            "controls": {
                "logging_interval_steps": int(legacy_output_controls["logging_interval_steps"]),
                "sample_surface_forces": bool(legacy_output_controls["sample_surface_forces"]),
                "surface_sample_forces": list(
                    legacy_output_controls.get("surface_sample_forces", ())
                ),
            },
        }
    legacy_fields = set(_FIELDS) - {
        "area",
        "relative_velocity",
        "bound_relative_velocity",
    }
    expected_fields = set(_FIELDS) if version >= 7 else legacy_fields
    if set(group) != {*expected_fields, "motion"}:
        raise ValueError("VLM restart fields are incomplete or unknown")
    for name in expected_fields:
        expected = getattr(vlm.lattice, name).to_numpy()[: vlm.lattice.n_panels]
        value = np.asarray(group[name])
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid VLM restart field {name}")
    expected_motion = {str(i) for i in range(len(vlm.surfaces))}
    if set(group["motion"]) != expected_motion:
        raise ValueError("VLM restart motion surface count does not match")
    for index, (_, kinematics) in enumerate(vlm.surfaces.values()):
        surface = group["motion"][str(index)]
        expected_names = {name for name in _MOTION_FIELDS if hasattr(kinematics, name)}
        if set(surface) != expected_names:
            raise ValueError("VLM restart motion fields do not match")
        for name in expected_names:
            value = np.asarray(surface[name])
            if value.shape != np.shape(getattr(kinematics, name)) or not np.isfinite(value).all():
                raise ValueError(f"Invalid VLM restart motion {name}")
    for name in ("time", "reference_speed", "reference_velocity"):
        if name not in group.attrs or not np.isfinite(group.attrs[name]).all():
            raise ValueError(f"Invalid VLM restart attribute {name}")
    if "force_density" in group.attrs and (
        not np.isfinite(group.attrs["force_density"]) or group.attrs["force_density"] <= 0.0
    ):
        raise ValueError("Invalid VLM restart force density")
    return identity_migration


def restore_vlm_restart(vlm, group):
    """Restore previously validated continuation data, invalidating derived solver caches."""
    present_fields = set(group)
    for name in _FIELDS:
        if name not in present_fields:
            continue
        field = getattr(vlm.lattice, name)
        values = field.to_numpy()
        values[: vlm.lattice.n_panels] = group[name][:]
        field.from_numpy(values)
    if "relative_velocity" not in present_fields:
        velocity = vlm.lattice.velocity.to_numpy()
        kinematic = vlm.lattice.kinematic_velocity.to_numpy()
        velocity[: vlm.lattice.n_panels] -= kinematic[: vlm.lattice.n_panels]
        vlm.lattice.relative_velocity.from_numpy(velocity)
    if "bound_relative_velocity" not in present_fields:
        bound = vlm.lattice.bound_vortex_velocity.to_numpy()
        kinematic = vlm.lattice.bound_kinematic_velocity.to_numpy()
        bound[: vlm.lattice.n_panels] -= kinematic[: vlm.lattice.n_panels]
        vlm.lattice.bound_relative_velocity.from_numpy(bound)
    for index, (_, kinematics) in enumerate(vlm.surfaces.values()):
        for name, value in group["motion"][str(index)].items():
            setattr(kinematics, name, np.asarray(value))
    vlm._solved = bool(group.attrs["solved"])
    vlm._coupled_mode = bool(group.attrs["coupled_mode"])
    vlm._current_time = float(group.attrs["time"])
    vlm.lattice.reference_speed = float(group.attrs["reference_speed"])
    if "force_density" in group.attrs:
        vlm._force_density = float(group.attrs["force_density"])
        vlm.lattice.force_density = vlm._force_density
    else:
        # A legacy checkpoint did not persist the density used for its
        # dimensional loads.  Never let a prior live solve leak that unknown
        # provenance into the restored state; compute_forces falls back to the
        # configured VLM density when this attribute is absent.
        if hasattr(vlm, "_force_density"):
            del vlm._force_density
        vlm.lattice.force_density = None
    vlm._last_reference_velocity = np.asarray(group.attrs["reference_velocity"])
    vlm._aerodynamic_influence_coefficient_computed = False
    vlm._linear_solver_instance = None
    vlm._bound_transport_ready = False
    if vlm._solved:
        vlm._last_forces = vlm.compute_forces(vlm.density, vlm._last_reference_velocity)
