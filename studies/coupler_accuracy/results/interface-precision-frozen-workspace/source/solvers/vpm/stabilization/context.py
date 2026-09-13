"""Explicit capabilities supplied to VPM stabilization workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..config.stabilization import StabilizationConfig


class ParticleMutationPort(Protocol):
    """The only particle mutations stabilization workers may request.

    Implementations own validation, capacity checks, state-revision updates,
    and removal accounting.  Stabilization workers should use this narrow
    interface instead of reaching into a particle container directly.
    """

    def replace(self, **properties: object) -> None:
        """Replace the complete active particle state from named arrays."""

        ...

    def set_properties(self, **properties: object) -> None:
        """Update selected per-particle fields without changing population."""

        ...

    def remove_by_bounds(self, bounds: list, *, invert_selection: bool = False) -> int:
        """Remove particles inside or outside an axis-aligned bounds list."""

        ...


class _ParticleMutationOwner(Protocol):
    """Concrete solver operations used by :class:`SolverParticleMutations`."""

    np_dtype: np.dtype
    _particles_removed_this_step: int
    _vortex_strength_removed_this_step: np.ndarray

    def replace_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        kinematic_viscosity: np.ndarray | None = None,
        eddy_viscosity: np.ndarray | None = None,
        group_id: np.ndarray | None = None,
        zone_id: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
        strain_rate: np.ndarray | None = None,
        report_removal: bool = True,
    ) -> None:
        """Replace the solver's active particle prefix with validated arrays."""

        ...

    def set_particles_properties(self, **properties: object) -> None:
        """Update selected aligned particle fields in place."""

        ...

    def remove_particles_by_bounds(self, bounds: list, invert_selection: bool = False) -> int:
        """Remove particles inside or outside a six-value axis-aligned box."""

        ...


@dataclass(frozen=True)
class SolverParticleMutations:
    """Adapter exposing the solver's approved particle-mutation operations."""

    owner: _ParticleMutationOwner
    state: StabilizationStepState

    def replace(self, **properties: object) -> None:
        """Delegate an atomic particle-state replacement to the solver."""
        self.owner.replace_vortex_particles(**properties)
        self._sync_removal_accounting()

    def set_properties(self, **properties: object) -> None:
        """Delegate a validated particle-property update to the solver."""
        self.owner.set_particles_properties(**properties)

    def remove_by_bounds(self, bounds: list, *, invert_selection: bool = False) -> int:
        """Delegate bounded particle removal and synchronize accounting."""
        removed = self.owner.remove_particles_by_bounds(bounds, invert_selection=invert_selection)
        self._sync_removal_accounting()
        return removed

    def _sync_removal_accounting(self) -> None:
        self.state.particles_removed = int(self.owner._particles_removed_this_step)
        self.state.vortex_strength_removed = np.asarray(
            self.owner._vortex_strength_removed_this_step, dtype=self.owner.np_dtype
        ).copy()


@dataclass
class StabilizationStepState:
    """Mutable, per-step stabilization bookkeeping owned by the coordinator."""

    step: int
    time: float
    time_step_size: float
    particles_removed: int = 0
    vortex_strength_removed: np.ndarray | None = None
    domain_bounds_enforced: bool = False


@dataclass
class StabilizationMetrics:
    """Latest scalar diagnostics available to stabilization workers."""

    kinetic_energy_rate: float = 0.0
    viscous_kinetic_energy_rate: float = 0.0


@dataclass(frozen=True)
class StabilizationContext:
    """Narrow typed interface between ``VPMSolver`` and stabilization workers."""

    particles: object
    physics: object
    field_diagnostics: object
    config: StabilizationConfig
    compute_dtype: object
    np_dtype: object
    flow_model: str

    state: StabilizationStepState
    mutations: ParticleMutationPort
    metrics: StabilizationMetrics
