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


def _schedule_metadata(schedule: object | None) -> dict[str, Any] | None:
    """Return only serializable, user-visible schedule controls."""
    if schedule is None:
        return None
    result: dict[str, Any] = {
        "type": type(schedule).__name__,
    }
    for name in ("interval", "first_step", "start_time", "initial", "is_final_only"):
        if hasattr(schedule, name):
            value = getattr(schedule, name)
            if isinstance(value, np.generic):
                value = value.item()
            result[name] = value
    return result


def _array_metadata(value: np.ndarray) -> dict[str, Any]:
    """Describe an array without expanding large particle configurations."""
    array = np.asarray(value)
    result: dict[str, Any] = {
        "type": "numpy.ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size <= 64:
        result["values"] = _metadata_value(array.tolist())
    else:
        contiguous = np.ascontiguousarray(array)
        result["sha256"] = hashlib.sha256(contiguous.tobytes()).hexdigest()
    return result


def _metadata_value(value: object) -> Any:
    """Convert construction values, including unbounded limits, to strict JSON.

    Infinite configuration bounds use the strings ``Infinity`` and
    ``-Infinity``. NaN remains invalid rather than hiding an invalid input.
    """
    if isinstance(value, np.ndarray):
        return _array_metadata(value)
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _metadata_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        result = {"type": type(value).__name__}
        for item in fields(value):
            result[item.name] = _metadata_value(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_metadata_value(item) for item in value), key=repr)
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
            result[name] = _metadata_value(getattr(value, name))
    return result


def _sampler_metadata(sampler: object) -> dict[str, Any]:
    """Describe a sampler without serializing its live callable state."""
    result: dict[str, Any] = {
        "type": type(sampler).__name__,
        "file_name": getattr(sampler, "file_name", None),
        "schedule": _schedule_metadata(getattr(sampler, "schedule", None)),
        "initial": bool(getattr(sampler, "initial", False)),
    }
    for name, value in getattr(sampler, "__dict__", {}).items():
        if name.startswith("_") or name in {"schedule", "file_name", "grid_points"}:
            continue
        if callable(value):
            continue
        result[name] = _metadata_value(value)
    return result


def _case_configuration(solver: Any) -> dict[str, Any]:
    """Return generic construction parameters shared by every VPM case."""
    case = solver.case
    run = case.run
    backup = case.backup
    samplers = case.samplers
    numerics = _metadata_value(case.numerics)
    vlm = getattr(solver, "vlm_solver", None)
    if vlm is not None:
        from ..boundary_elements.vlm.geometry.surface_io import surface_to_dict
        from ..boundary_elements.vlm.solver.restart import (
            restart_configuration_hash,
            restart_physics_hash,
        )

        # Keep the full continuation configuration and the numerical configuration
        # explicit; output controls participate only in the former.
        numerics["vlm"]["configuration_hash"] = getattr(
            vlm, "_restart_configuration_hash", restart_configuration_hash(vlm)
        )
        numerics["vlm"]["physics_hash"] = getattr(
            vlm,
            "_restart_physics_hash",
            restart_physics_hash(vlm),
        )

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
            "state_limit_action": str(run.state_limit_action),
            "wall_time_limit_seconds": run.wall_time_limit_seconds,
            "runtime_compute_device": run.runtime_compute_device,
        },
        "backup": {
            "interval_steps": int(backup.interval_steps),
            "directory": str(backup.directory),
            "log_directory": str(backup.log_directory),
        },
        "samplers": {
            "directory": samplers.directory,
            "items": [_sampler_metadata(sampler) for sampler in samplers.samples],
        },
        "initial_conditions": [
            _metadata_value(initial_condition) for initial_condition in case.initial_conditions
        ],
        "initial_weak_particle_percent": float(case.initial_weak_particle_percent),
    }


def _json_default(value: object) -> object:
    """Serialize NumPy scalars while rejecting hidden live runtime objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in VPM metadata")


def build_metadata(solver: Any, *, status: str | None = None) -> dict[str, Any]:
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
        Run status to include. ``None`` omits the ``run_status`` member.

    Returns
    -------
    dict[str, object]
        JSON-compatible metadata using schema version 2.
    """
    restored_particle_count = getattr(solver, "_restart_particle_count", None)
    particle_count = (
        int(restored_particle_count)
        if restored_particle_count is not None
        else int(getattr(getattr(solver, "particles", None), "n_particles_total", 0))
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
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
            "n_particles_total": particle_count,
        },
    }
    if status is not None:
        metadata["run_status"] = {"status": str(status)}
    restart_details = getattr(solver, "_restart_details", None)
    if restart_details is not None:
        metadata["restart"] = _metadata_value(restart_details)
    runtime_override = getattr(solver, "_runtime_compute_device_override", None)
    if runtime_override is not None:
        metadata["runtime"] = {
            "configured_compute_device": solver.setup.compute_device,
            "requested_compute_device": runtime_override,
            "effective_compute_device": getattr(solver, "_backend_name", solver.compute_device),
            "numerical_settings_unchanged": True,
        }
    return metadata


def write_metadata(solver: Any, path: str | Path, *, status: str) -> Path:
    """Atomically write the solver's universal VPM metadata.

    ``VPMSolver`` passes its configured backup directory, so metadata and the
    numerical state it describes have one solver and one destination.

    Parameters
    ----------
    solver : VPMSolver
        Initialized solver to describe.
    path : str or pathlib.Path
        Destination metadata file.
    status : str
        Current solver run status.

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
    metadata = json.dumps(
        build_metadata(solver, status=status),
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
            stream.write(metadata + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination
