"""Gaussian infinite-image tail error_bounds; no solver-settings integration.

The bounds are float64 evaluations with one safety factor (``float_bounds``),
valid under round-to-nearest with or without flush-to-zero.
Snapshots retain original immutable source bytes for explicit exact validation.
Their error_bounds do not validate finite-image interpolation or GPU fields.
"""

from .error_bounds import (
    PreparedTailSource,
    QueryTailBound,
    SourceValueMismatchError,
    prepare_tail_source,
    query_tail_bound,
    validate_source_values,
)

__all__ = [
    "PreparedTailSource",
    "QueryTailBound",
    "SourceValueMismatchError",
    "prepare_tail_source",
    "query_tail_bound",
    "validate_source_values",
]
