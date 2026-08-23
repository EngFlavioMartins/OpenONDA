"""Small, dependency-free helpers for canonical serialized field maps.

Writers should call these helpers immediately before crossing a file or
inter-solver boundary. They deliberately do not rename data: noncanonical
names are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .physical_fields import SCHEMA_VERSION, validate_serialized_field_name


def validate_field_map(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a shallow copy of an outgoing field mapping."""
    result = dict(fields)
    for name in result:
        if not isinstance(name, str):
            raise TypeError(f"serialized field names must be strings, got {type(name).__name__}")
        validate_serialized_field_name(name)
    result["physical_field_schema_version"] = SCHEMA_VERSION
    return result


def schema_metadata(**metadata: Any) -> dict[str, Any]:
    """Return canonical schema metadata merged with caller-owned metadata."""
    result = dict(metadata)
    result["physical_field_schema_version"] = SCHEMA_VERSION
    return result


__all__ = ["schema_metadata", "validate_field_map"]
