"""Serializable VPM state models and state-management helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def cached_particle_property(func: F) -> F:
    """Cache an expensive particle property for one particle-state revision.

    Particle fields can be changed several times while the solver clock stays
    fixed (for example by a stabilization worker or a coupled wake update).
    A step number is therefore not a cache key.  ``Particles.touch_state``
    publishes a new monotone revision whenever a source field changes.
    """
    cache_name = f"_{func.__name__}_cache"
    cache_revision_name = f"_{func.__name__}_cache_revision"

    @wraps(func)
    def wrapper(self, use_cache: bool = True):
        cache_valid = (
            use_cache
            and getattr(self, cache_revision_name, None) == self.state_revision
            and hasattr(self, cache_name)
        )
        if not cache_valid:
            setattr(self, cache_name, func(self))
            setattr(self, cache_revision_name, self.state_revision)
        return getattr(self, cache_name)

    return wrapper  # type: ignore[return-value]


def set_flow_model(
    solver: Any,
    flow_model: str,
) -> None:
    """Set the solver flow model and concise physical description."""
    descriptions = {
        "DNS": "DNS ::: (ω·∇)u + ν∇²ω",
        "LES": "LES ::: (ω·∇)u + (ν+νt)∇²ω",
        "INVISCID": "INV ::: (ω·∇)u",
    }
    solver.flow_model = flow_model
    if flow_model in descriptions:
        solver.flow_model_description = descriptions[flow_model]
