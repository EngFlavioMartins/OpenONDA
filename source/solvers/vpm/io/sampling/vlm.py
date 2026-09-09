"""VLM surface fields dispatched through the scientific output manager."""

from dataclasses import dataclass
from pathlib import Path

from .schedule import OutputSchedule


@dataclass(frozen=True)
class VLMSampler:
    """Write solved lattice geometry, circulation, velocity and loads to samples.

    The regular output manager owns the sample cadence, atomic writes and PVD
    index. ``Backup`` independently writes sparse surface companions beside
    numerical restart files, even when no ``VLMSampler`` is configured.
    """

    schedule: OutputSchedule | None = None
    file_name: str = "vlm"
    initial: bool = False
    vtk_extension: str = ".vtp"

    def save_vtp(self, solver, filepath: Path, time: float | None = None) -> None:
        """Export the accepted lattice as VTK PolyData."""
        if solver.vlm_solver is None:
            raise ValueError("VLMSampler requires a configured VLM solver")
        solver.vlm_solver.save_results(str(filepath.with_suffix("")), time=time)
