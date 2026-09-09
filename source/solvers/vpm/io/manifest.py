"""Solver-owned metadata for VPM runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _schedule_identity(schedule: object | None) -> dict[str, Any] | None:
    """Return only serializable, user-visible schedule controls."""
    if schedule is None:
        return None
    result: dict[str, Any] = {
        "type": type(schedule).__name__,
    }
    for name in ("interval", "initial", "is_final_only", "at_end"):
        if hasattr(schedule, name):
            value = getattr(schedule, name)
            if isinstance(value, np.generic):
                value = value.item()
            result[name] = value
    return result


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    """Describe an array without expanding large particle configurations."""
    array = np.asarray(value)
    result: dict[str, Any] = {
        "type": "numpy.ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size <= 64:
        result["values"] = _manifest_value(array.tolist())
    else:
        contiguous = np.ascontiguousarray(array)
        result["sha256"] = hashlib.sha256(contiguous.tobytes()).hexdigest()
    return result


def _manifest_value(value: object) -> Any:
    """Convert construction values, including unbounded limits, to strict JSON.

    Infinite configuration bounds use the strings ``Infinity`` and
    ``-Infinity``. NaN remains invalid rather than hiding an invalid input.
    """
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, np.generic):
        return _manifest_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _manifest_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        result = {"type": type(value).__name__}
        for item in fields(value):
            result[item.name] = _manifest_value(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_manifest_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_manifest_value(item) for item in value), key=repr)
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    result = {"type": type(value).__name__}
    for name in (
        "method",
        "stretching_scheme",
        "theta",
        "multipole_order",
        "tolerance",
        "expansion_order",
        "sort_particle_targets",
        "traversal_block_dim",
        "velocity",
        "final_velocity",
        "acceleration_time",
        "start_time",
        "angular_speed",
        "axis",
        "rotation_centre",
        "initial_velocity",
        "acceleration",
        "amplitude",
        "frequency",
        "phase",
        "direction",
        "kinematics_components",
    ):
        if hasattr(value, name):
            result[name] = _manifest_value(getattr(value, name))
    return result


def _sampler_identity(sampler: object) -> dict[str, Any]:
    """Describe a sampler without serializing its live callable state."""
    result: dict[str, Any] = {
        "type": type(sampler).__name__,
        "file_name": getattr(sampler, "file_name", None),
        "schedule": _schedule_identity(getattr(sampler, "schedule", None)),
        "initial": bool(getattr(sampler, "initial", False)),
    }
    for name, value in getattr(sampler, "__dict__", {}).items():
        if name.startswith("_") or name in {"schedule", "file_name", "grid_points"}:
            continue
        if callable(value):
            continue
        result[name] = _manifest_value(value)
    return result


def _case_configuration(solver: Any) -> dict[str, Any]:
    """Return generic construction parameters shared by every VPM case."""
    case = solver.case
    run = case.run
    backup = case.backup
    samplers = case.samplers
    numerics = _manifest_value(case.numerics)
    vlm = getattr(solver, "vlm_solver", None)
    if vlm is not None:
        from ..boundary_elements.vlm.geometry.surface_io import surface_to_dict

        for record, (geometry, _) in zip(
            numerics["vlm"]["surfaces"], vlm.surfaces.values(), strict=True
        ):
            # Capture the geometry actually loaded by the solver. Re-reading
            # its source file here would let later input edits alter this run.
            record["geometry"] = surface_to_dict(geometry)
    return {
        "numerics": numerics,
        "run": {
            "steps": int(run.steps),
            "initial_samples": bool(run.initial_samples),
            "final_backup": bool(run.final_backup),
            "health_limit_action": str(run.health_limit_action),
            "wall_time_limit_seconds": run.wall_time_limit_seconds,
        },
        "backup": {
            "interval_steps": int(backup.interval_steps),
            "directory": str(backup.directory),
            "log_directory": str(backup.log_directory),
        },
        "samplers": {
            "directory": samplers.directory,
            "items": [_sampler_identity(sampler) for sampler in samplers.samples],
        },
        "initial_conditions": [
            _manifest_value(initial_condition) for initial_condition in case.initial_conditions
        ],
        "initial_weak_particle_percent": float(case.initial_weak_particle_percent),
    }


def _json_default(value: object) -> object:
    """Serialize NumPy scalars while rejecting hidden live runtime objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in VPM metadata")


def build_manifest(solver: Any, *, status: str | None = None) -> dict[str, Any]:
    """Collect universal VPM construction and accepted-state metadata.

    The schema deliberately contains no tutorial-specific experiment labels,
    resume signatures, or termination explanations. Large explicit NumPy
    arrays are represented by shape, dtype, and a content hash rather than
    duplicated.

    Parameters
    ----------
    solver : VPMSolver
        Initialized solver whose immutable case configuration and latest
        accepted state are described. Live backend objects are represented by
        their public construction controls only.
    status : str or None, optional
        Lifecycle status to include. ``None`` omits the ``lifecycle`` member.

    Returns
    -------
    dict[str, object]
        JSON-compatible metadata using schema version 1.
    """
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "solver": "VPM",
        "case_name": solver.case.name,
        "configuration": _case_configuration(solver),
        "state": {
            "initial_step": int(getattr(solver, "_run_initial_step", 0)),
            "initial_time": float(getattr(solver, "_run_initial_time", 0.0)),
            "step": int(solver.step),
            "time": float(solver.time),
            "requested_steps": int(solver.case.run.steps),
            "initial_n_particles_total": int(getattr(solver, "_initial_n_particles_total", 0)),
            "n_particles_total": int(
                getattr(getattr(solver, "particles", None), "n_particles_total", 0)
            ),
        },
    }
    if status is not None:
        manifest["lifecycle"] = {"status": str(status)}
    return manifest


def write_manifest(solver: Any, path: str | Path, *, status: str) -> Path:
    """Atomically write the solver's universal VPM metadata.

    ``VPMSolver`` passes its configured backup directory, so metadata and the
    numerical state it describes have one owner and one destination.

    Parameters
    ----------
    solver : VPMSolver
        Initialized solver to describe.
    path : str or pathlib.Path
        Destination metadata file.
    status : str
        Current solver lifecycle status.

    Returns
    -------
    pathlib.Path
        Destination path after the atomic replacement succeeds.

    Raises
    ------
    TypeError
        If a public configuration value cannot be represented as JSON.
    OSError
        If the destination directory or metadata file cannot be written.

    Side Effects
    ------------
    Creates the destination directory when needed and atomically replaces the
    metadata file.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_manifest(solver, status=status),
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination
