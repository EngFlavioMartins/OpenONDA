"""Output of one FVM solver: diagnostics, visualization frames, run metadata and histories."""

from __future__ import annotations

import csv
from dataclasses import asdict
import errno
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.clock import time_tolerance
from openonda.core.file_output import append_jsonl, rewind_csv, rewind_jsonl
from openonda.core.metadata import write_run_metadata
from openonda.core.parallel import collective_phase
from openonda.core.solution_layout import collection_path, component_directory
from openonda.core.status import RunStatus
from openonda.core.vtk_output import add_frame, rewind_collection

from .async_output import BufferedVTKWriter
from .partitioned import write_partition_vtu
from .vtk_exporter import VTKExporter, mesh_cell_fields

if TYPE_CHECKING:
    from ..core.solver import FVMSolver


def mesh_counts(mesh_data: dict) -> dict[str, int]:
    """Return the numbers of cells, faces and points of ``mesh_data``."""
    n_points = mesh_data.get("n_points")
    if n_points is None:
        n_points = len(mesh_data["vertex_position"])
    return {
        "n_cells": int(mesh_data["n_cells"]),
        "n_faces": int(mesh_data["n_faces"]),
        "n_points": int(n_points),
    }


class SolverIO:
    """Write the run's output files and rewind them on a restart.

    Rank 0 writes ``solution/diagnostics.jsonl``, the run metadata and the
    history files; the visualization frames are written by every rank of a
    partitioned run. A full disk stops the diagnostics with a warning; other
    file errors propagate.

    Attributes
    ----------
    vtk_exporter : VTKExporter or None
        Exporter of the visualization frames, created with the first frame.
    buffered_vtk_writer : BufferedVTKWriter or None
        Writer of asynchronous frames, created with the first frame.
    last_vtk_state : tuple or None
        ``(step, time, state revision)`` of the last frame written.
    run_metadata_written : bool
        Whether ``fvm_metadata.json`` holds the state of the closed run.
    initial_run_selected : bool
        Whether the run starts from the initial state, so a history is cleared.
    """

    def __init__(self, solver: FVMSolver) -> None:
        self.solver = solver
        self._diagnostics_stopped = False
        self.vtk_exporter: VTKExporter | None = None
        self.buffered_vtk_writer: BufferedVTKWriter | None = None
        self.last_vtk_state: tuple[int, float, int] | None = None
        self.run_metadata_written = False
        self.initial_run_selected = False
        self._rewound_histories: set[Path] = set()

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
        self.flush()

    def flush(self) -> None:
        """Wait for queued visualization frames and surface writer errors."""
        if self.buffered_vtk_writer is not None:
            self.buffered_vtk_writer.flush()

    def close(self) -> None:
        """Finish the queued visualization frames and stop the writer."""
        if self.buffered_vtk_writer is not None:
            self.buffered_vtk_writer.close()

    def write_run_metadata(self, status: RunStatus | str) -> Path:
        """Write ``solution/fvm_metadata.json`` with the configuration and state; return its path."""
        solver = self.solver
        destination = solver.solution_dir / "fvm_metadata.json"
        with collective_phase(solver.parallel.comm, "solver metadata"):
            if solver.parallel.is_root:
                mesh_data = solver.mesh_data
                # A partitioned run holds only the owned and halo cells of rank 0 here;
                # the counts of the whole mesh were kept when it was distributed.
                counts = mesh_data.get("_global_mesh_counts") or mesh_counts(mesh_data)
                write_run_metadata(
                    destination,
                    solver="fvm",
                    case_name=solver._configuration.case_name,
                    status=RunStatus(status),
                    configuration=solver._configuration,
                    sections={
                        "mesh": {
                            **counts,
                            "mesh_generation": mesh_data.get("mesh_generation"),
                        },
                        "state": {"step": int(solver.step), "time": float(solver.time)},
                    },
                )
        self.run_metadata_written = True
        return destination

    def write_vtk(self) -> None:
        """Write the accepted state as a visualization frame; collective under MPI."""
        solver = self.solver
        with collective_phase(solver.parallel.comm, "visualization output"):
            if solver.parallel.is_root or solver.parallel.is_partitioned:
                self._write_vtk_local()
        self.last_vtk_state = (solver.step, solver.time, solver._state_revision)

    def _write_vtk_local(self) -> None:
        """Write this rank's part of the visualization frame."""
        solver = self.solver
        frame_dir = component_directory(solver.solution_dir, "fvm")
        frame_dir.mkdir(parents=True, exist_ok=True)
        filename = frame_dir / f"fvm_{solver.step:06d}.vtu"
        collection = collection_path(solver.solution_dir, "fvm")
        output = solver._configuration.output

        fields = {
            "velocity": solver.velocity,
            "kinematic_pressure": solver.kinematic_pressure,
            "courant_number": solver._courant_field(solver._accepted_time_step_size),
            "vorticity": solver._vorticity_field(),
        }
        fields.update(mesh_cell_fields(solver.mesh_data, solver.geo_data["cell_volume"]))
        if solver.eddy_viscosity is not None:
            fields["eddy_viscosity"] = solver.eddy_viscosity

        if solver.parallel.is_partitioned:
            n_local = solver.mesh_data["n_cells"]
            if self.vtk_exporter is None:
                export_mesh = solver.mesh_data["_visualization_mesh"]
                self.vtk_exporter = VTKExporter(export_mesh, output)
            frame = write_partition_vtu(
                frame_dir,
                filename.stem,
                solver.mesh_data,
                solver.parallel.partition,
                {name: np.asarray(values)[:n_local] for name, values in fields.items()},
                solver.parallel.comm,
                output=output,
                exporter=self.vtk_exporter,
            )
            if solver.parallel.is_root:
                add_frame(collection, solver.time, frame)
                solver.logger.event(f"Output written: {Path(frame).name}", immediate=True)
            return

        if output.asynchronous:
            if self.buffered_vtk_writer is None:
                self.buffered_vtk_writer = BufferedVTKWriter(solver.mesh_data, collection, output)
            self.buffered_vtk_writer.submit(filename, solver.time, fields)
            action = "queued"
        else:
            if self.vtk_exporter is None:
                self.vtk_exporter = VTKExporter(solver.mesh_data, output)
            self.vtk_exporter.export(str(filename), fields)
            add_frame(collection, solver.time, filename)
            action = "written"
        solver.logger.event(f"Output {action}: {filename.name}", immediate=True)

    def restore_history(self, filename: str | Path) -> bool:
        """Rewind a case CSV history to the accepted time; return whether it holds that time."""
        solver = self.solver
        current = False
        with collective_phase(solver.parallel.comm, "history time alignment"):
            if solver.parallel.is_root:
                path = Path(filename)
                if not path.is_absolute():
                    path = solver.solution_dir / path
                if self.initial_run_selected:
                    if path not in self._rewound_histories:
                        rewind_csv(path, -1.0e100)
                        self._rewound_histories.add(path)
                else:
                    rewind_csv(path, solver.time)
                if path.is_file():
                    with path.open(newline="", encoding="utf-8") as stream:
                        current = any(
                            abs(float(row["time"]) - solver.time) <= time_tolerance(solver.time)
                            for row in csv.DictReader(stream)
                        )
        return solver.parallel.bcast(current, root=0)
