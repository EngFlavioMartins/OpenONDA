"""Solver-owned metadata for FVM runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    """Describe an array without copying a full cell field into metadata."""
    array = np.asarray(value)
    result: dict[str, Any] = {
        "type": "numpy.ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size <= 64:
        result["values"] = array.tolist()
    else:
        result["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return result


def _manifest_value(value: object) -> Any:
    """Convert a construction value to concise, stable JSON data."""
    if isinstance(value, np.ndarray):
        return _array_identity(value)
    if isinstance(value, np.generic):
        return value.item()
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
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return {"type": type(value).__name__}


def build_manifest(solver: Any, *, status: str | None = None) -> dict[str, Any]:
    """Collect universal FVM construction and accepted-state metadata.

    The schema deliberately omits tutorial-specific labels and failure prose.
    Large initial fields are represented by shape, dtype, and a content hash.

    Parameters
    ----------
    solver : FVMSolver
        Initialized solver whose resolved setup, mesh summary, and latest
        accepted state are described.
    status : str or None, optional
        Lifecycle status to include. ``None`` omits the ``lifecycle`` member.

    Returns
    -------
    dict[str, object]
        JSON-compatible metadata using schema version 1.
    """
    from source.solvers.fvm.sampling.base import sampler_to_dict

    config = getattr(solver, "_resolved_setup", solver.setup)
    configuration = _manifest_value(config)
    if config.samplers:
        configuration["samplers"] = [
            _manifest_value(sampler_to_dict(sampler)) for sampler in config.samplers
        ]
    n_points = solver.mesh_data.get("n_points")
    if n_points is None:
        n_points = len(solver.mesh_data["vertex_position"])
    manifest = {
        "schema_version": 1,
        "solver": "FVM",
        "case_name": config.case_name,
        "configuration": configuration,
        "mesh": {
            "n_cells": int(solver.mesh_data["n_cells"]),
            "n_faces": int(solver.mesh_data["n_faces"]),
            "n_points": int(n_points),
            "provenance": _manifest_value(solver.mesh_data.get("provenance")),
        },
        "state": {
            "step": int(solver.step),
            "time": float(solver.time),
        },
    }
    if status is not None:
        manifest["lifecycle"] = {"status": str(status)}
    return manifest


def write_manifest(solver: Any, path: str | Path, *, status: str | None = None) -> Path:
    """Atomically write the solver's universal FVM metadata.

    Parameters
    ----------
    solver : FVMSolver
        Initialized solver to describe.
    path : str or pathlib.Path
        Destination metadata file.
    status : str or None, optional
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

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} in FVM metadata")

    payload = json.dumps(
        build_manifest(solver, status=status),
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=json_default,
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
