"""Gaussian infinite-image tail certificates; no solver-policy integration.

Callers must provide a round-to-nearest, gradual-underflow environment for
preparation and queries. These functions never switch FENV controls; a caller
such as a Taichi host may require a separately owned save/restore scope.
Snapshots retain original immutable source bytes for explicit exact admission.
Their certificates do not certify finite-image interpolation or GPU fields.
"""

from .certificate import (
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
