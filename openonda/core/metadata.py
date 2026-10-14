"""Run metadata written by every solver to ``solution/<solver>_metadata.json``.

The metadata records the solver, the case name, the run status and the
complete case configuration, followed by solver-specific sections such as
the mesh size or the particle count. Large arrays are recorded by shape,
data type and content hash instead of their values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .file_output import write_json_atomic
from .status import RunStatus
from .version import __version__

__all__ = [
    "METADATA_SCHEMA_VERSION",
    "metadata_value",
    "write_run_metadata",
]

METADATA_SCHEMA_VERSION = 3
"""Version of the metadata layout written by :func:`write_run_metadata`."""

_SHARED_KEYS = frozenset(
    ("case_name", "configuration", "openonda_version", "run_status", "schema_version", "solver")
)
_MAX_LISTED_ARRAY_SIZE = 64


def _array_value(array: np.ndarray) -> dict[str, Any]:
    """Describe an array by its values when small, otherwise by its hash."""
    result: dict[str, Any] = {
        "type": "numpy.ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size <= _MAX_LISTED_ARRAY_SIZE:
        result["values"] = metadata_value(array.tolist())
    else:
        result["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return result


def metadata_value(value: object) -> Any:
    """Convert a configuration value to JSON-compatible metadata.

    Parameters
    ----------
    value : object
        Configuration value: a dataclass, schedule, sampler, mapping,
        sequence, set, NumPy array or scalar, enumeration, path, or a plain
        object such as a user-defined initial-condition builder.

    Returns
    -------
    object
        JSON-compatible value. Dataclasses become a mapping with a ``type``
        member and their fields; objects with ``to_dict`` or ``config_dict``
        use it; other objects record their type and public attributes.
        Arrays with more than 64 values are recorded by shape, data type
        and SHA-256 hash. Infinite floats become ``"Infinity"`` or
        ``"-Infinity"``.

    Raises
    ------
    ValueError
        If a float is NaN.
    """
    return _metadata_value(value, set())


def _metadata_value(value: object, active: set[int]) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("Metadata cannot record NaN")
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, np.ndarray):
        return _array_value(value)
    if isinstance(value, np.generic):
        return _metadata_value(value.item(), active)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _metadata_value(value.value, active)
    if id(value) in active:
        return {"type": type(value).__name__}
    active.add(id(value))
    try:
        if is_dataclass(value) and not isinstance(value, type):
            result: dict[str, Any] = {"type": type(value).__name__}
            for item in fields(value):
                result[item.name] = _metadata_value(getattr(value, item.name), active)
            return result
        if isinstance(value, Mapping):
            return {str(key): _metadata_value(item, active) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [_metadata_value(item, active) for item in value]
        if isinstance(value, set | frozenset):
            return sorted((_metadata_value(item, active) for item in value), key=repr)
        if isinstance(value, type):
            return {"type": value.__name__}
        for method in ("to_dict", "config_dict"):
            if callable(getattr(value, method, None)):
                converted = _metadata_value(getattr(value, method)(), active)
                if isinstance(converted, dict):
                    return {"type": type(value).__name__, **converted}
                return {"type": type(value).__name__, "value": converted}
        result = {"type": type(value).__name__}
        for name, item in getattr(value, "__dict__", {}).items():
            if not name.startswith("_") and not callable(item):
                result[name] = _metadata_value(item, active)
        return result
    finally:
        active.discard(id(value))


def write_run_metadata(
    path: str | Path,
    *,
    solver: str,
    case_name: str,
    status: RunStatus | str,
    configuration: object,
    sections: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write the metadata of a solver run.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination, normally ``solution/<solver>_metadata.json``.
    solver : str
        Solver name, for example ``"fvm"``, ``"vpm"`` or ``"coupler"``.
    case_name : str
        Name of the case.
    status : RunStatus or str
        Current run status; a string must be a :class:`RunStatus` value.
    configuration : object
        Case configuration, converted with :func:`metadata_value`.
    sections : mapping of str to object or None, default=None
        Solver-specific top-level sections, for example ``{"state": {"step":
        10, "time": 0.5}, "mesh": {...}}``, converted with
        :func:`metadata_value`.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    ValueError
        If ``status`` is not a run status, a section name repeats a shared
        key, or a value is NaN.
    TypeError
        If a value cannot be written as JSON.
    """
    document: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "openonda_version": __version__,
        "solver": str(solver),
        "case_name": str(case_name),
        "run_status": {"status": str(RunStatus(status))},
        "configuration": metadata_value(configuration),
    }
    for name, section in (sections or {}).items():
        if name in _SHARED_KEYS:
            raise ValueError(f"Metadata section {name!r} repeats a shared key")
        document[str(name)] = metadata_value(section)
    return write_json_atomic(path, document)
