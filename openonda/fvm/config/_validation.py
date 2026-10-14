"""Scalar, vector and schedule checks shared by the FVM configuration classes."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly


def finite_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    """Return ``value`` as a finite float inside the given range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = normalized <= minimum if strict_minimum else normalized < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return normalized


def strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    """Return an integer count, rejecting booleans and fractional values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def vector_or_field(name: str, value: Any) -> tuple[float, float, float] | np.ndarray:
    """Return a detached uniform ``(3,)`` vector or ``(n, 3)`` field."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) and (array.ndim != 2 or array.shape[1:] != (3,)):
        raise ValueError(f"{name} must have shape (3,) or (n, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if array.ndim == 1:
        return tuple(float(component) for component in array)
    copied = array.copy()
    copied.setflags(write=False)
    return copied


def check_schedule(name: str, schedule: object, *, optional: bool = False) -> None:
    """Reject anything other than a shared output schedule."""
    if schedule is None and optional:
        return
    if not isinstance(schedule, EverySteps | EveryTime | FinalOnly):
        allowed = "EverySteps, EveryTime, FinalOnly" + (" or None" if optional else "")
        raise TypeError(f"{name} must be {allowed}")


def lowercase_choice(name: str, value: Any, choices: set[str] | frozenset[str]) -> str:
    """Return a string option after checking it against its lowercase spellings."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}; got {value!r}")
    return str(value)
