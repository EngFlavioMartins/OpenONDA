"""One output precision setting for every field OpenONDA writes to disk.

Write precision is independent of the precision a solver computes in. A run
may integrate in ``f64`` and still write ``f32`` output, because a stored
field is read by ParaView and by post-processing, not by the time integrator.

Three levels are available. ``f64`` and ``f32`` are stored in containers of
the matching width. ``f16`` rounds each value to half precision and stores the
result in a ``float32`` container: neither VTK XML nor other visualization
readers understand a 16-bit float, so a native half array would produce files
ParaView cannot open. Rounding still pays, because the discarded mantissa
bytes become zeros that the deflate and byte-shuffle filters remove almost
entirely.

Integer fields remain unchanged at every precision: particle and cell identifiers
are exact quantities and are never rounded.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

WritePrecision = Literal["f16", "f32", "f64"]

DEFAULT_WRITE_PRECISION: WritePrecision = "f32"

_STORAGE_DTYPE: dict[str, np.dtype] = {
    "f16": np.dtype(np.float32),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def validate_write_precision(value: str, *, field_name: str = "write_precision") -> str:
    """Return *value* when it names a supported write precision."""
    if value not in _STORAGE_DTYPE:
        raise ValueError(f"{field_name} must be one of {', '.join(_STORAGE_DTYPE)}, got {value!r}")
    return value


def storage_dtype(precision: str) -> np.dtype:
    """Return the on-disk dtype a write precision is stored in."""
    validate_write_precision(precision)
    return _STORAGE_DTYPE[precision]


def cast_for_write(values: ArrayLike, precision: str) -> np.ndarray:
    """Return *values* as a contiguous array at the requested write precision.

    Non-floating arrays pass through unchanged. A value beyond the range of
    ``f32`` is written as an infinity, as a NaN is written as a NaN; VTK and
    HDF5 store both. Half precision is clamped to its finite limits, so
    that a diverging field still produces a file ParaView colour maps can
    scale.
    """
    validate_write_precision(precision)
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array)

    if precision == "f16":
        limit = np.finfo(np.float16)
        with np.errstate(over="ignore"):
            half = array.astype(np.float16, order="C")
        single = half.astype(np.float32)
        np.clip(single, float(limit.min), float(limit.max), out=single)
        return single

    target = _STORAGE_DTYPE[precision]
    if array.dtype == target:
        return np.ascontiguousarray(array)
    with np.errstate(over="ignore"):
        return array.astype(target, order="C")
