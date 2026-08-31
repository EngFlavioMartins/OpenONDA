"""Explicit capabilities supplied to VPM stabilization workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..config.stabilization import StabilizationConfig


class ParticleMutationPort(Protocol):
    """The only particle mutations stabilization workers may request."""

    def replace(self, **properties: Any) -> None: ...
    def set_properties(self, **properties: Any) -> None: ...
    def remove_by_bounds(self, bounds: list, *, invert_selection: bool = False) -> int: ...


@dataclass(frozen=True)
class SolverParticleMutations:
    """Adapter exposing the solver's approved particle-mutation operations."""

    owner: Any
    state: StabilizationStepState

    def replace(self, **properties: Any) -> None:
        self.owner.replace_vortex_particles(**properties)
        self._sync_removal_accounting()

    def set_properties(self, **properties: Any) -> None:
        self.owner.set_particles_properties(**properties)

    def remove_by_bounds(self, bounds: list, *, invert_selection: bool = False) -> int:
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

    particles: Any
    physics: Any
    field_diagnostics: Any
    config: StabilizationConfig
    compute_dtype: Any
    np_dtype: Any
    flow_model: str

    state: StabilizationStepState
    mutations: ParticleMutationPort
    metrics: StabilizationMetrics
