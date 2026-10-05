"""Exact VLM continuation state inside the owning VPM numerical backup."""

import hashlib
import inspect
import json
import marshal
from types import CodeType

import numpy as np

from ....io.metadata import _metadata_value

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
_OUTPUT_CONTROL_FIELDS = ("logging_interval_steps", "sample_surface_forces")


def _portable_code(code):
    """Retain executable callback code while ignoring its source location."""
    constants = tuple(
        _portable_code(value) if isinstance(value, CodeType) else value for value in code.co_consts
    )
    return code.replace(co_filename="<motion>", co_firstlineno=0, co_consts=constants)


def _restart_controls(vlm, *, include_output_controls=True):
    """Return the deterministic VLM controls used by restart hashes."""
    controls = _metadata_value(vlm.setup)
    # The setup describes user choices; this explicit field settings also
    # describes the geometric source/trace/transport operators so a
    # checkpoint cannot silently continue under a different field model.
    field_settings = getattr(vlm, "field_settings", None)
    if field_settings is not None and hasattr(field_settings, "as_dict"):
        controls["field_settings"] = field_settings.as_dict()
    for surface in controls["surfaces"]:
        # Geometry and reference values are hashed below; a moved case retains
        # the same numerical configuration without depending on an absolute filename.
        surface.pop("surface")
        if not include_output_controls:
            surface.pop("sample_forces", None)
    if not include_output_controls:
        for name in _OUTPUT_CONTROL_FIELDS:
            controls.pop(name, None)
    return controls


def _configuration_hash(vlm, controls):
    """Hash controls, callbacks and generated geometry for one configuration hash."""
    controls["geometry_references"] = getattr(
        vlm,
        "_restart_geometry_references",
        _metadata_value(vlm.aircraft.refs),
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
                captured = _metadata_value({**closure.globals, **closure.nonlocals})
                digest.update(json.dumps(captured, sort_keys=True, allow_nan=False).encode())
    for name in ("panel_corner_position", "normal"):
        digest.update(getattr(vlm.lattice, name).to_numpy()[: vlm.lattice.n_panels].tobytes())
    return digest.hexdigest()


def restart_configuration_hash(vlm):
    """Hash declared physics and output controls plus generated initial geometry."""
    return _configuration_hash(vlm, _restart_controls(vlm))


def restart_physics_hash(vlm):
    """Hash only VLM controls that can change solved or emitted physics.

    Logging cadence and per-surface sampling do not change evolution and are
    omitted from the physics hash. The full configuration hash separately records them.
    """
    return _configuration_hash(vlm, _restart_controls(vlm, include_output_controls=False))


def write_vlm_restart(vlm, group):
    """Store lattice fields and mutable rigid-motion state without reducing precision."""
    group.attrs["version"] = 8
    group.attrs["configuration_hash"] = vlm._restart_configuration_hash
    group.attrs["physics_hash"] = getattr(
        vlm,
        "_restart_physics_hash",
        restart_physics_hash(vlm),
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


def validate_vlm_restart(vlm, group):
    """Validate the current VLM backup schema and numerical configuration before mutation."""
    if group is None:
        raise ValueError("VPM backup is missing VLM continuation state")
    version = int(group.attrs.get("version", -1))
    if version != 8:
        raise ValueError("Incompatible VLM restart version; start a new run with this solver")
    expected_attributes = {
        "version",
        "configuration_hash",
        "solved",
        "coupled_mode",
        "time",
        "reference_speed",
        "reference_velocity",
        "physics_hash",
    }
    allowed_attributes = (expected_attributes, expected_attributes | {"force_density"})
    if set(group.attrs) not in allowed_attributes:
        raise ValueError("VLM restart attributes are incomplete or unknown")
    for name in ("solved", "coupled_mode"):
        if group.attrs[name] not in (False, True):
            raise ValueError(f"Invalid VLM restart flag {name}")
    if np.shape(group.attrs["reference_velocity"]) != (3,):
        raise ValueError("Invalid VLM restart reference velocity")
    expected_physics_hash = getattr(vlm, "_restart_physics_hash", None)
    if expected_physics_hash is None:
        expected_physics_hash = restart_physics_hash(vlm)
    if group.attrs.get("physics_hash") != expected_physics_hash:
        raise ValueError("VLM restart physics configuration does not match this solver")
    if group.attrs.get("configuration_hash") != vlm._restart_configuration_hash:
        raise ValueError("VLM restart geometry or configuration does not match this solver")
    expected_fields = set(_FIELDS)
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


def restore_vlm_restart(vlm, group):
    """Restore previously validated continuation data, invalidating derived solver caches."""
    for name in _FIELDS:
        field = getattr(vlm.lattice, name)
        values = field.to_numpy()
        values[: vlm.lattice.n_panels] = group[name][:]
        field.from_numpy(values)
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
        # An unsolved state has no dimensional force evaluation to restore.
        if hasattr(vlm, "_force_density"):
            del vlm._force_density
        vlm.lattice.force_density = None
    vlm._last_reference_velocity = np.asarray(group.attrs["reference_velocity"])
    vlm._aerodynamic_influence_coefficient_computed = False
    vlm._linear_solver_instance = None
    vlm._bound_transport_ready = False
    if vlm._solved:
        vlm._last_forces = vlm.compute_forces(vlm.density, vlm._last_reference_velocity)
