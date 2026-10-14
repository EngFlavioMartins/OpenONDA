"""Particle and vortex-lattice visualization frames of the VPM solver.

Particle frames are ``solution/vpm/vpm_{step:06d}.vtu``, listed with their
solution times in ``solution/vpm.pvd``; an attached vortex lattice writes
``solution/vlm/vlm_{step:06d}.vtp`` and ``solution/vlm.pvd`` at the same
states. Frames are for visualization only; restarts read the HDF5 backups.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.solution_layout import collection_path, component_directory
from openonda.core.vtk_output import read_collection, write_collection, write_vtk_dataset
from openonda.core.write_precision import cast_for_write

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

__all__ = ["write_particle_frame", "write_solution_frames"]

_VECTOR_FIELDS = ("velocity", "vortex_strength", "vorticity")
_SCALAR_FIELDS = (
    "core_radius",
    "particle_volume",
    "kinematic_viscosity",
    "eddy_viscosity",
    "effective_viscosity",
    "group_id",
    "zone_id",
)


def write_particle_frame(solver: VPMSolver, path: str | Path) -> Path:
    """Write the particles of the current state as a VTK unstructured grid.

    One ``VTK_POLY_VERTEX`` cell holds every particle. Point data are the
    particle fields in SI units at ``Output.precision``; the field data
    ``time`` and ``TimeValue`` hold the solution time in s.
    """
    from vtk import VTK_POLY_VERTEX, vtkCellArray, vtkPoints, vtkUnstructuredGrid
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    output = solver.case.output
    particles = solver.particles
    n_particles = int(particles.n_particles_total)
    time = float(solver.time)
    if not np.isfinite(time):
        raise ValueError("VPM frame time must be finite")
    position = cast_for_write(particles.position_cpu(use_cache=False), output.precision)

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(position), deep=True))
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points)
    if n_particles:
        cells = vtkCellArray()
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.zeros(1, dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.arange(n_particles, dtype=np.int64), deep=True),
        )
        grid.SetCells(numpy_to_vtk(np.array([VTK_POLY_VERTEX], dtype=np.uint8), deep=True), cells)

    for name in (*_VECTOR_FIELDS, *_SCALAR_FIELDS):
        values = getattr(particles, f"{name}_cpu")(use_cache=False)
        array = numpy_to_vtk(
            np.ascontiguousarray(cast_for_write(values, output.precision)), deep=True
        )
        array.SetName(name)
        grid.GetPointData().AddArray(array)
    for name in ("time", "TimeValue"):
        field = numpy_to_vtk(np.array([time], dtype=np.float64), deep=True)
        field.SetName(name)
        grid.GetFieldData().AddArray(field)
    return write_vtk_dataset(grid, Path(path), compression=output.compression)


def _add_to_collection(solution_dir: Path, component: str, frame: Path, time: float) -> None:
    collection = collection_path(solution_dir, component)
    entries = read_collection(collection)
    write_collection(collection, [*entries, (time, frame.relative_to(solution_dir).as_posix())])


def write_solution_frames(solver: VPMSolver) -> Path:
    """Write the particle frame and lattice surface of the current state.

    The frames are added to ``solution/vpm.pvd`` and ``solution/vlm.pvd``.

    Returns
    -------
    pathlib.Path
        The particle frame.
    """
    solution_dir = solver.solution_dir
    time = float(solver.time)
    particle_frame = component_directory(solution_dir, "vpm") / f"vpm_{solver.step:06d}.vtu"
    write_particle_frame(solver, particle_frame)
    _add_to_collection(solution_dir, "vpm", particle_frame, time)
    vlm = solver.vlm_solver
    if vlm is not None:
        surface_frame = component_directory(solution_dir, "vlm") / f"vlm_{solver.step:06d}.vtp"
        surface_frame.parent.mkdir(parents=True, exist_ok=True)
        vlm.save_results(str(surface_frame.with_suffix("")), time=time)
        _add_to_collection(solution_dir, "vlm", surface_frame, time)
    return particle_frame
