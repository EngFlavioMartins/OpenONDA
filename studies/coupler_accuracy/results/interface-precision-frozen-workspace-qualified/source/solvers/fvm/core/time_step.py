"""Solver-owned selection of transient FVM time-step sizes."""

from __future__ import annotations

import math

from ..config.types import MaximumCourantTimeStep

_MAXIMUM_GROWTH_FACTOR = 1.2
_GROWTH_DAMPING = 0.1


def maximum_courant_time_step_size(
    current_time_step_size: float,
    current_maximum_courant_number: float,
    control: MaximumCourantTimeStep,
) -> float:
    """Return the next OpenFOAM-style maximum-Courant time-step estimate.

    The measured Courant number is proportional to the time-step size for a
    fixed face-flux field.  Consequently ``control.maximum / current`` is the
    factor that would land exactly on the target.  Following OpenFOAM's
    ``setDeltaT`` policy, reductions are immediate, while growth is limited by
    both ``1 + 0.1 * factor`` and a hard factor of ``1.2``.  The optional
    ``maximum_time_step_size`` is applied last.

    A zero Courant number contains no finite CFL estimate; in that case the
    same twenty-percent growth limit is used.  Inputs are validated here so
    restart corruption or an invalid runtime state cannot silently select an
    unusable step.
    """
    current_time_step_size = float(current_time_step_size)
    current_maximum_courant_number = float(current_maximum_courant_number)
    if not math.isfinite(current_time_step_size) or current_time_step_size <= 0.0:
        raise ValueError("current_time_step_size must be finite and positive")
    if not math.isfinite(current_maximum_courant_number) or current_maximum_courant_number < 0.0:
        raise ValueError("current_maximum_courant_number must be finite and non-negative")

    if current_maximum_courant_number > 0.0:
        target_factor = control.maximum / current_maximum_courant_number
        adjustment_factor = min(
            target_factor,
            1.0 + _GROWTH_DAMPING * target_factor,
            _MAXIMUM_GROWTH_FACTOR,
        )
    else:
        adjustment_factor = _MAXIMUM_GROWTH_FACTOR

    selected = current_time_step_size * adjustment_factor
    if control.maximum_time_step_size is not None:
        selected = min(selected, control.maximum_time_step_size)
    if not math.isfinite(selected) or selected <= 0.0:
        raise FloatingPointError("Maximum-Courant control selected an invalid time step")
    return selected


__all__ = ["maximum_courant_time_step_size"]
