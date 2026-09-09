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
    "panel_force",
    "unsteady_panel_force",
    "unsteady_pressure_jump_coefficient",
    "panel_moment_correction",
    "pressure_coefficient",
    "leading_edge_suction_parameter",
)
_MOTION_FIELDS = ("current_position", "current_orientation", "rotation_centre")


def _portable_code(code):
    """Retain executable callback identity while ignoring its source location."""
    constants = tuple(
        _portable_code(value) if isinstance(value, CodeType) else value for value in code.co_consts
    )
    return code.replace(co_filename="<motion>", co_firstlineno=0, co_consts=constants)


def restart_identity(vlm):
    """Hash declared controls and the generated initial geometry before motion."""
    controls = _manifest_value(vlm.setup)
    # Omission retains the previous shedding rule and restart identity.
    if controls["wake_core_overlap"] is None:
        controls.pop("wake_core_overlap")
    for surface in controls["surfaces"]:
        # Geometry and reference values are hashed below; a moved case retains
        # the same numerical identity without depending on an absolute filename.
        surface.pop("surface")
    controls["geometry_references"] = _manifest_value(vlm.aircraft.refs)
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


def write_vlm_restart(vlm, group):
    """Store lattice fields and mutable rigid-motion state without reducing precision."""
    group.attrs["version"] = 6
    group.attrs["identity"] = vlm._restart_identity
    group.attrs["solved"] = vlm._solved
    group.attrs["coupled_mode"] = vlm._coupled_mode
    group.attrs["time"] = vlm._current_time if vlm._current_time is not None else 0.0
    group.attrs["reference_speed"] = vlm.lattice.reference_speed
    group.attrs["reference_velocity"] = getattr(vlm, "_last_reference_velocity", np.zeros(3))
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
    """Validate all VLM state before any live particle or lattice fields change."""
    if group is None:
        raise ValueError("VPM backup is missing VLM continuation state")
    expected_attributes = {
        "version",
        "identity",
        "solved",
        "coupled_mode",
        "time",
        "reference_speed",
        "reference_velocity",
    }
    if set(group.attrs) != expected_attributes:
        raise ValueError("VLM restart attributes are incomplete or unknown")
    for name in ("solved", "coupled_mode"):
        if group.attrs[name] not in (False, True):
            raise ValueError(f"Invalid VLM restart flag {name}")
    if np.shape(group.attrs["reference_velocity"]) != (3,):
        raise ValueError("Invalid VLM restart reference velocity")
    if group.attrs.get("version") != 6:
        raise ValueError(
            "VLM restart uses an incompatible wake/force-state version; "
            "start a new run with this solver"
        )
    if group.attrs.get("identity") != vlm._restart_identity:
        raise ValueError("VLM restart geometry or configuration does not match this solver")
    if set(group) != {*_FIELDS, "motion"}:
        raise ValueError("VLM restart fields are incomplete or unknown")
    for name in _FIELDS:
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
    vlm._last_reference_velocity = np.asarray(group.attrs["reference_velocity"])
    vlm._aerodynamic_influence_coefficient_computed = False
    vlm._linear_solver_instance = None
    vlm._bound_transport_ready = False
    if vlm._solved:
        vlm._last_forces = vlm.compute_forces(vlm.density, vlm._last_reference_velocity)
