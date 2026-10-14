"""Step diagnostics and restart rewind of the FVM output histories."""

from __future__ import annotations

from dataclasses import asdict
import errno
from pathlib import Path
from typing import TYPE_CHECKING

from openonda.core.file_output import append_jsonl, rewind_csv, rewind_jsonl
from openonda.core.solution_layout import collection_path
from openonda.core.vtk_output import rewind_collection

if TYPE_CHECKING:
    from ..core.solver import FVMSolver


class SolverIO:
    """Write ``solution/diagnostics.jsonl`` and rewind the output histories.

    Only rank 0 writes. A full disk stops the diagnostics with a warning;
    other file errors propagate.
    """

    def __init__(self, solver: FVMSolver) -> None:
        self.solver = solver
        self._diagnostics_stopped = False

    def write_step_diagnostics(self) -> None:
        """Append the diagnostics of the accepted step as one JSON line."""
        solver = self.solver
        record = solver.last_diagnostics
        if not solver.parallel.is_root or record is None or self._diagnostics_stopped:
            return
        path = solver.solution_dir / "diagnostics.jsonl"
        try:
            append_jsonl(path, asdict(record))
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise
            self._diagnostics_stopped = True
            solver.logger.warning(f"Diagnostic output stopped because the disk is full: {path}")

    def rewind_histories(self, time: float) -> None:
        """Delete the output recorded after the physical time ``time`` in s.

        The sampler CSV files and collections, the diagnostics and
        performance journals and the visualization collection with its frames
        are trimmed. Unrelated files in ``samples/`` are kept.
        """
        solver = self.solver
        if not solver.parallel.is_root:
            return
        samples = Path(solver.samples_dir)
        for name in {sampler.name for sampler in solver._samplers}:
            rewind_csv(samples / f"{name}.csv", time)
            rewind_collection(samples / f"{name}.pvd", time)
        solution = Path(solver.solution_dir)
        rewind_jsonl(solution / "diagnostics.jsonl", time)
        rewind_jsonl(solution / "performance.jsonl", time)
        rewind_collection(collection_path(solution, "fvm"), time)
        solver._frame_collection = None
        if solver._buffered_vtk_writer is not None:
            solver._buffered_vtk_writer.rewind(time)
        solver._sample_collections.clear()
