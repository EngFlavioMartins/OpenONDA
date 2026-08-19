"""Explicit capability context for the stabilization workers.

The stabilization package must not reach into the solver as a service
locator.  :class:`StabilizationContext` is the only thing the workers are
allowed to see: it carries the particle container, the narrow physics
capabilities regularization needs (grid-based diffusion and flow-integral
evaluation), the stabilization configuration, the precision, and the few
live scalar values and mutation entry points the workers schedule against.

It is a plain data holder, not a dependency-injection framework.  Mutable
step state (``step``, ``time``, ``time_step_size``) is read
through callables so the workers always observe the current step; particle
mutation is limited to the explicit VPMSolver entry points listed here, so
``StabilizationManager`` has no unrestricted view of ``VPMSolver`` internals.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config.types import StabilizationConfig


@dataclass(frozen=True)
class StabilizationContext:
    """The genuine dependencies of the stabilization subsystem.

    Attributes:
        particles: The live particle container (read/write via its own API).
        physics: Physics engine; used only for the grid-based diffusion
            (DVH) machinery of conservative regularization.
        field_diagnostics: Flow-integral evaluation used to audit the
            conservative-regularization candidates.
        config: The stabilization configuration.
        compute_dtype: Taichi compute dtype of the solver.
        np_dtype: Matching NumPy dtype.
        flow_model: ``"POTENTIAL"``, ``"VORTEX"``, or ``"LES"`` (constant).
        step: Callable returning the current step index.
        time: Callable returning the current physical time.
        time_step_size: Callable returning the current time-step size.
        replace_vortex_particles: VPMSolver entry point replacing the whole
            particle cloud.
        set_particles_properties: VPMSolver entry point mutating particle
            properties in place.
        remove_particles_by_bounds: VPMSolver entry point removing particles
            outside a domain box.
        particles_removed: Read the step's removed-particle counter.
        set_particles_removed: Write the step's removed-particle counter.
        circulation_removed: Read the step's removed-circulation vector.
        set_circulation_removed: Write the step's removed-circulation vector.
        domain_bounds_enforced: Whether the current regeneration already
            guaranteed the configured retention box.
        set_domain_bounds_enforced: Update that per-step guarantee.
    """

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
    circulation_removed: Callable[[], np.ndarray]
    set_circulation_removed: Callable[[np.ndarray], None]
    domain_bounds_enforced: Callable[[], bool]
    set_domain_bounds_enforced: Callable[[bool], None]
