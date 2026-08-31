"""Canonical online flow-integral sampler."""

from __future__ import annotations

from pathlib import Path

from ..io.logging import Logging
from ..io.sampling.schedule import SamplingSchedule


class FlowIntegralsSampler:
    """Compute, report, and append the VPM integral diagnostics."""

    requires_flow_integrals = True

    def __init__(
        self,
        *,
        schedule: SamplingSchedule | None = None,
        file_name: str = "flow_integrals",
    ) -> None:
        if not file_name:
            raise ValueError("FlowIntegralsSampler file_name must not be empty")
        self.schedule = schedule
        self.file_name = file_name

    def save_csv(
        self,
        solver,
        path: Path,
        *,
        time: float,
        step: int | None,
    ) -> None:
        del time, step
        Logging.flow_diagnostics(solver)
        if solver.turbulence_model is not None:
            Logging.les_diagnostics(solver)
        solver.io.export_flow_integrals_csv(solver, path)
