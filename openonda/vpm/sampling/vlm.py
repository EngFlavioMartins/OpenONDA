"""Vortex-lattice surface frames of a standalone VLM sampler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openonda.core.sampling import Sampler
from openonda.core.schedules import EverySteps, Schedule

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

__all__ = ["VLMSampler"]


class VLMSampler(Sampler):
    """Write the lattice geometry, circulation, velocity and loads as VTK frames.

    A VPM case with an attached lattice writes the surface with every
    particle frame and rejects this sampler; it serves a lattice whose
    surface frames need their own schedule.

    Parameters
    ----------
    file_name : str, default='vlm'
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
    initial : bool, default=True
        Output arguments of :class:`~openonda.core.sampling.Sampler`.
    """

    vtk_extension = ".vtp"

    def __init__(
        self,
        *,
        file_name: str = "vlm",
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)

    def write_vtk(self, solver: VPMSolver, path: Path, time: float) -> Path:
        """Write the solved lattice at solution time ``time`` (s) as VTK PolyData."""
        if solver.vlm_solver is None:
            raise ValueError("VLMSampler requires a configured VLM solver")
        solver.vlm_solver.save_results(str(Path(path).with_suffix("")), time=time)
        return Path(path)
