"""Explicit capabilities supplied to VPM stabilization workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config.stabilization import StabilizationConfig


@dataclass(frozen=True)
class StabilizationContext:
    """Narrow interface between ``VPMSolver`` and stabilization workers."""

    particles: Any
    physics: Any
    field_diagnostics: Any
    config: StabilizationConfig
    compute_dtype: Any
    np_dtype: Any
    flow_model: str

    step: Callable[[], int]
    time: Callable[[], float]
    time_step_size: Callable[[], float]

    replace_vortex_particles: Callable[..., None]
    set_particles_properties: Callable[..., None]
    remove_particles_by_bounds: Callable[..., None]

    particles_removed: Callable[[], int]
    set_particles_removed: Callable[[int], None]

    vortex_strength_removed: Callable[[], np.ndarray]
    set_vortex_strength_removed: Callable[[np.ndarray], None]

    domain_bounds_enforced: Callable[[], bool]
    set_domain_bounds_enforced: Callable[[bool], None]
