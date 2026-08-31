"""Internal, deterministic identities for VPM numerical configurations.

This module deliberately produces a restart identity, not a user-facing case
serialization format.  Live output handlers, panel objects, VLM surfaces, and
kinematics are not reconstructible configuration data and must never be
silently represented as such.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .setup import VPMSetup


def _canonical_value(value: Any) -> Any:
    """Convert numerical value objects to JSON-compatible deterministic values."""
    if is_dataclass(value):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def numerical_configuration(setup: VPMSetup) -> dict[str, Any]:
    """Return the resolved settings that determine VPM particle evolution.

    The mapping is internal restart evidence only. It intentionally
    excludes clocks, output plans, and live coupled objects; callers cannot use
    it to recreate a solver configuration.
    """
    return {
        "advection": _canonical_value(setup.advection),
        "axisymmetric_no_swirl_axis": setup.axisymmetric_no_swirl_axis,
        "compute_device": setup.compute_device,
        "cutoff_radius_factor": setup.cutoff_radius_factor,
        "domain_bounds": _canonical_value(setup.domain_bounds),
        "health_limits": _canonical_value(setup.health_limits),
        "max_n_particles": setup.max_n_particles,
        "particle_kernel": setup.particle_kernel,
        "precision": setup.precision,
        "random_seed": setup.random_seed,
        "stabilization": _canonical_value(setup.stabilization),
        "stretching": _canonical_value(setup.stretching),
        "time_integration": setup.time_integration,
        "time_step_size": setup.time_step_size,
        "turbulence": _canonical_value(setup.turbulence),
        "velocity": _canonical_value(setup.velocity),
        "viscous": _canonical_value(setup.viscous),
    }
