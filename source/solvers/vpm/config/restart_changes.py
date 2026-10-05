"""Exact configuration changes when loading a VPM backup.

Each allowed scalar path applies only to that value. Adding or replacing a
setting group requires the exact stored and current values. Backup schema,
precision, field and clock checks always apply.
"""

from collections.abc import Collection, Mapping
from enum import Enum
import json
import re
from typing import Any

from .restart import _configuration_mismatches, restart_configuration_values


class _MissingConfigurationValue(Enum):
    MISSING = "missing configuration value"


MISSING_CONFIGURATION_VALUE = _MissingConfigurationValue.MISSING

_PATH = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\[\d+\])*(?:\.[A-Za-z_][A-Za-z_0-9]*(?:\[\d+\])*)*\Z")
_PART = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)|\[(\d+)\]")


def _value_at(configuration: object, path: str) -> object:
    value = configuration
    for key, index in _PART.findall(path):
        if key:
            if not isinstance(value, dict) or key not in value:
                return MISSING_CONFIGURATION_VALUE
            value = value[key]
        else:
            if not isinstance(value, list) or int(index) >= len(value):
                return MISSING_CONFIGURATION_VALUE
            value = value[int(index)]
    return value


def _configuration_json(value: object) -> str | _MissingConfigurationValue:
    if value is MISSING_CONFIGURATION_VALUE:
        return MISSING_CONFIGURATION_VALUE
    # Reject non-JSON values and NaN instead of accepting equality through
    # Python's permissive comparison (notably True == 1).
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _evidence(value: object) -> dict[str, Any]:
    if value is MISSING_CONFIGURATION_VALUE:
        return {"present": False}
    return {"present": True, "value": json.loads(_configuration_json(value))}


def _validate_changes(
    current: dict[str, Any],
    stored: dict[str, Any],
    mismatches: set[str],
    *,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
    protected_paths: Collection[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Check each allowed configuration difference."""
    if isinstance(allowed_config_differences, str | bytes) or not isinstance(
        allowed_config_differences, Collection
    ):
        raise TypeError("allowed_config_differences must be a collection of exact paths")
    paths = tuple(allowed_config_differences)
    if any(type(path) is not str or not _PATH.fullmatch(path) for path in paths):
        raise ValueError("configuration permissions require exact non-wildcard dotted paths")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate configuration permission paths")
    allowed = set(paths)
    if "time_step_size" in allowed and "time_step_size" in protected_paths:
        raise ValueError(
            "time_step_size requires the explicit time_step_size continuation override"
        )
    if expected_config_differences is None:
        expectations = {}
    elif not isinstance(expected_config_differences, Mapping):
        raise TypeError("expected_config_differences must map paths to (stored, current) values")
    else:
        expectations = dict(expected_config_differences)
    if any(type(path) is not str for path in expectations) or set(expectations) - allowed:
        raise ValueError("configuration expectations require matching allowed paths")
    frozen_expectations = {}
    for path, values in expectations.items():
        if not isinstance(values, tuple | list) or len(values) != 2:
            raise ValueError(f"configuration expectation {path!r} requires (stored, current)")
        frozen_expectations[path] = tuple(_configuration_json(value) for value in values)

    unused = allowed - mismatches
    if unused:
        raise ValueError(
            "configuration permission is not an exact changed path: " + ", ".join(sorted(unused))
        )
    unpermitted = mismatches - allowed
    if unpermitted:
        raise ValueError("numerical configuration mismatch at " + ", ".join(sorted(unpermitted)))

    evidence = []
    for path in sorted(allowed):
        old, new = _value_at(stored, path), _value_at(current, path)
        structural = any(
            value is MISSING_CONFIGURATION_VALUE or isinstance(value, dict | list)
            for value in (old, new)
        )
        if structural and path not in frozen_expectations:
            raise ValueError(
                f"structured configuration difference {path!r} requires exact stored/current expectations"
            )
        if path in frozen_expectations and frozen_expectations[path] != (
            _configuration_json(old),
            _configuration_json(new),
        ):
            raise ValueError(f"configuration expectation mismatch at {path}")
        evidence.append({"path": path, "stored": _evidence(old), "current": _evidence(new)})
    return tuple(evidence)


def validate_configuration_changes(
    expected: dict[str, Any],
    found: dict[str, Any],
    *,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
    allow_time_step_size_mismatch: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Validate exact VPM permissions and return detached old/new evidence.

    Paths are relative to the VPM configuration, e.g. ``induction.theta``.
    Expectations are ``path: (stored_value, current_value)``. Truly missing
    keys use :data:`MISSING_CONFIGURATION_VALUE`, not ``None``. Every requested
    path must be an actual incompatible path; parent paths, wildcard paths,
    unused permissions and expectations without permission are rejected.
    ``time_step_size`` is reserved for the separate explicit step-size API.
    Execution placement and hard storage capacity are operational.
    """
    current = restart_configuration_values(expected)
    stored = restart_configuration_values(found)
    mismatches = set(_configuration_mismatches(current, stored))
    if allow_time_step_size_mismatch:
        mismatches.discard("time_step_size")
    return _validate_changes(
        current,
        stored,
        mismatches,
        allowed_config_differences=allowed_config_differences,
        expected_config_differences=expected_config_differences,
        protected_paths={"time_step_size"},
    )


def _exact_mismatches(current: object, stored: object, path: str = "") -> set[str]:
    if isinstance(current, dict) and isinstance(stored, dict):
        result = set()
        for key in sorted(set(current) | set(stored)):
            child = f"{path}.{key}" if path else key
            if key not in current or key not in stored:
                result.add(child)
            else:
                result.update(_exact_mismatches(current[key], stored[key], child))
        return result
    if isinstance(current, list) and isinstance(stored, list) and len(current) == len(stored):
        result = set()
        for index, (new, old) in enumerate(zip(current, stored, strict=True)):
            result.update(_exact_mismatches(new, old, f"{path}[{index}]"))
        return result
    return set() if _configuration_json(current) == _configuration_json(stored) else {path}


def validate_exact_configuration_changes(
    expected: dict[str, Any],
    found: dict[str, Any],
    *,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generic exact validation without operational exemptions."""
    return _validate_changes(
        expected,
        found,
        _exact_mismatches(expected, found),
        allowed_config_differences=allowed_config_differences,
        expected_config_differences=expected_config_differences,
    )


__all__ = [
    "MISSING_CONFIGURATION_VALUE",
    "validate_configuration_changes",
    "validate_exact_configuration_changes",
]
