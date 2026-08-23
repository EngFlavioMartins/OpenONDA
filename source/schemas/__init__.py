"""Canonical schemas shared by all OpenONDA solvers and serializers."""

from .physical_fields import (
    FIELD_REGISTRY,
    SCHEMA_VERSION,
    FieldDefinition,
    canonical_component_names,
    validate_field_contract,
    validate_field_name,
    validate_serialized_field_name,
)
from .serialization import schema_metadata, validate_field_map

__all__ = [
    "FIELD_REGISTRY",
    "SCHEMA_VERSION",
    "FieldDefinition",
    "canonical_component_names",
    "validate_field_contract",
    "validate_serialized_field_name",
    "validate_field_name",
    "schema_metadata",
    "validate_field_map",
]
