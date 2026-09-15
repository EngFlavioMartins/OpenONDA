"""Canonical online flow-integral sampler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..io.logging import Logging
from ..io.sampling.schedule import OutputSchedule

if TYPE_CHECKING:
    from ..core.solver import VPMSolver
    from ..io.sampler import SamplingContext


class FlowIntegralsSampler:
    """Append canonical global VPM invariants and energy diagnostics to CSV.

    Parameters
    ----------
    schedule : OutputSchedule or None, optional
        Accepted-state output cadence. ``None`` lets the output manager apply
        its default policy.
    file_name : str, default='flow_integrals'
        Non-empty basename below the case samples directory; ``.csv`` is added
        by framework-owned dispatch.
    initial : bool or None, default=None
        Sample the initial state as well as the regular cadence. ``None``
        retains a subclass's initial-state policy (normally disabled).

    Notes
    -----
    The owning output manager refreshes flow integrals before calling this
    sampler. Values include vector circulation (m³/s), impulses (m⁴/s and
    m⁵/s), energy per density (m⁵/s²), helicity (m⁴/s²), and enstrophy
    (m³/s²). Writes are side effects; particle state is read only.
    """

    requires_flow_integrals = True

    def __init__(
        self,
        *,
        schedule: OutputSchedule | None = None,
        file_name: str = "flow_integrals",
        initial: bool | None = None,
    ) -> None:
        """Validate and retain output cadence and basename without writing files."""
        if not file_name:
            raise ValueError("FlowIntegralsSampler file_name must not be empty")
        self.schedule = schedule
        self.file_name = file_name
        if initial is not None:
            self.initial = initial

    def save_csv(
        self,
        solver: VPMSolver,
        path: Path,
        *,
        time: float,
        step: int | None = None,
    ) -> None:
        """Append the canonical integral row for one accepted solver state.

        Parameters
        ----------
        solver : VPMSolver
            Solver whose refreshed integral properties are serialized.
        path : pathlib.Path
            CSV destination; parent-directory ownership belongs to the output
            manager.
        time : float
            Accepted physical time in s, retained for the common interface.
        step : int or None, optional
            Accepted step index, retained for the common interface.

        Notes
        -----
        Emits diagnostic log records and appends to ``path``. No particle field
        is modified.
        """
        del time, step
        Logging.flow_diagnostics(solver)
        if solver.turbulence_model is not None and Logging._routine_messages_enabled:
            solver.turbulence_model.update_turbulence_statistics(solver.particles)
            Logging.les_diagnostics(solver)
        solver.io.export_flow_integrals_csv(solver, path)

    def write(self, context: SamplingContext) -> None:
        """Write one restart-aware diagnostic event from typed runtime context."""
        self.save_csv(
            context.solver, context.output_directory / f"{self.file_name}.csv", time=context.time
        )
