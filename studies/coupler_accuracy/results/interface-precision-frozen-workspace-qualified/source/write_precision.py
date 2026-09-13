"""One precision policy for every field OpenONDA writes to disk.

Write precision is independent of the precision a solver computes in. A run
may integrate in ``f64`` and still publish ``f32`` output, because a stored
field is read by ParaView and by post-processing, not by the time integrator.

Three levels are available. ``f64`` and ``f32`` are stored in containers of
the matching width. ``f16`` rounds each value to half precision and stores the
result in a ``float32`` container: neither VTK XML nor the XDMF readers
understand a 16-bit float, so a native half array would produce files ParaView
cannot open. Rounding still pays, because the discarded mantissa bytes become
zeros that the deflate and byte-shuffle filters remove almost entirely.

Integer fields are identity, whatever the level: particle and cell identifiers
are exact quantities and are never rounded.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

WritePrecision = Literal["f16", "f32", "f64"]

WRITE_PRECISIONS: tuple[str, ...] = ("f16", "f32", "f64")

DEFAULT_WRITE_PRECISION: WritePrecision = "f32"

_STORAGE_DTYPE: dict[str, np.dtype] = {
    "f16": np.dtype(np.float32),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def validate_write_precision(value: str, *, field_name: str = "write_precision") -> str:
    """Return *value* when it names a supported write precision."""
    if value not in _STORAGE_DTYPE:
        raise ValueError(
            f"{field_name} must be one of {', '.join(WRITE_PRECISIONS)}, got {value!r}"
        )
    return value


def storage_dtype(precision: str) -> np.dtype:
    """Return the on-disk dtype a write precision is stored in."""
    validate_write_precision(precision)
    return _STORAGE_DTYPE[precision]


def cast_for_write(values, precision: str) -> np.ndarray:
    """Return *values* as a contiguous array at the requested write precision.

    Non-floating arrays pass through unchanged. Values outside the target
    range are clamped to its finite limits rather than becoming infinities,
    so a diverging field still produces a file a reader can open.
    """
    validate_write_precision(precision)
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array)

    if precision == "f16":
        limit = np.finfo(np.float16)
        rounded = np.clip(array, float(limit.min), float(limit.max)).astype(np.float16)
        return np.ascontiguousarray(rounded.astype(np.float32))

    target = _STORAGE_DTYPE[precision]
    if array.dtype == target:
        return np.ascontiguousarray(array)
    limit = np.finfo(target)
    clamped = np.clip(array, float(limit.min), float(limit.max))
    return np.ascontiguousarray(clamped.astype(target))


def cast_fields_for_write(fields: dict, precision: str) -> dict:
    """Return a new mapping with every floating field at the write precision."""
    return {name: cast_for_write(values, precision) for name, values in fields.items()}
