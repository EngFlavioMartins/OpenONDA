"""Internal, deterministic values for VPM numerical configurations.

This module produces a restart configuration, not a user-facing case
serialization format. Live output handlers, VLM surfaces, and
kinematics are not reconstructible configuration data and must never be
silently represented as such.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .case import Numerics


def configuration_value(value: Any) -> Any:
    """Convert a numerical setting to a JSON-compatible deterministic value."""
    if is_dataclass(value):
        return {item.name: configuration_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): configuration_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [configuration_value(item) for item in value]
    if isinstance(value, np.generic):
        return configuration_value(value.item())
    if isinstance(value, float) and math.isinf(value):
        # Infinite bounds are valid numerical controls, but not JSON numbers.
        return "Infinity" if value > 0 else "-Infinity"
    return value


def numerical_configuration(setup: Numerics) -> dict[str, Any]:
    """Return the resolved settings that determine VPM particle evolution.

    The mapping is internal restart evidence only. It
    excludes clocks, output plans, and live coupled objects; callers cannot use
    it to recreate a solver configuration. Infinite bounds are represented by
    the JSON strings ``Infinity`` and ``-Infinity``.
    """
    return {
        "domain_bounds": configuration_value(setup.domain_bounds),
        "state_limits": configuration_value(setup.state_limits),
        "max_n_particles": setup.max_n_particles,
        "particle_kernel": setup.particle_kernel,
        "precision": setup.precision,
        "random_seed": setup.random_seed,
        "stabilization": configuration_value(setup.stabilization),
        "integrator": configuration_value(setup.integrator),
        "induction": setup.induction.configuration(),
        "time_step_size": setup.time_step_size,
        "turbulence": configuration_value(setup.turbulence),
        "viscous": configuration_value(setup.viscous),
    }
