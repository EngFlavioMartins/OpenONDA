"""Complete, detached fields for application analyses, with one output owner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundarySnapshot:
    """Physical boundary faces in global face order (no processor faces)."""

    face_centre: np.ndarray
    velocity: np.ndarray
    kinematic_pressure: np.ndarray


@dataclass(frozen=True)
class AnalysisSnapshot:
    """Global interior-cell fields and physical boundary values in SI units.

    Arrays are detached from the running solver. Analyses may read these fields
    and write files without knowing how the numerical domain is partitioned.
    """

    step: int
    time: float
    cell_centre: np.ndarray
    cell_volume: np.ndarray
    velocity: np.ndarray
    kinematic_pressure: np.ndarray
    vorticity: np.ndarray
    boundaries: dict[str, BoundarySnapshot]
    max_courant_number: float
    max_continuity_error: float


def evaluate(solver, callback, *args, **kwargs):
    """Gather owned fields, evaluate once, and share the result or failure."""
    from ..fields.diagnostics import compute_continuity_error

    parallel = solver.parallel
    mesh = solver.mesh_data
    n_cells = mesh["n_cells"]
    n_interior = mesh["n_interior_faces"]
    n_owned = parallel.n_owned if parallel.is_partitioned else n_cells
    payload = None
    failure = None
    try:
        boundaries = {}
        for patch in solver.boundaries:
            start, count = patch["start_face"], patch["n_faces"]
            faces = np.arange(start, start + count)
            ghosts = n_cells + faces - n_interior
            ids = mesh["global_face_id"][faces] if parallel.is_partitioned else faces
            boundaries[patch["name"]] = (
                ids,
                solver.geo_data["face_centre"][faces],
                solver.velocity[ghosts],
                solver.kinematic_pressure[ghosts],
            )
        continuity = compute_continuity_error(
            solver.volumetric_face_flux,
            mesh,
            solver.geo_data,
        )[:n_owned] / (solver.geo_data["cell_volume"][:n_owned] + 1e-30)
        ids = parallel.partition.owned_global_ids if parallel.is_partitioned else np.arange(n_cells)
        payload = (
            ids,
            solver.geo_data["cell_centre"][:n_owned].copy(),
            solver.geo_data["cell_volume"][:n_owned].copy(),
            solver.velocity[:n_owned].copy(),
            solver.kinematic_pressure[:n_owned].copy(),
            solver._vorticity_field()[:n_owned].copy(),
            boundaries,
            float(np.max(np.abs(continuity), initial=0.0)),
        )
    except BaseException as error:
        failure = error
    solver._collective_io_failure(failure, "analysis field preparation")
    parts = parallel.comm.gather(payload, root=0) if parallel.is_parallel else [payload]
    result = None
    failure = None
    if parallel.is_root:
        try:
            if not parallel.is_partitioned:
                parts = parts[:1]
            cell_ids = np.concatenate([part[0] for part in parts])
            order = np.argsort(cell_ids)
            if not np.array_equal(cell_ids[order], np.arange(len(cell_ids))):
                raise RuntimeError("Analysis requires exactly one owner of every global cell")
            fields = [np.concatenate([part[i] for part in parts])[order] for i in range(1, 6)]
            boundaries = {}
            for name in dict.fromkeys(name for part in parts for name in part[6]):
                patches = [part[6][name] for part in parts if name in part[6]]
                face_ids = np.concatenate([patch[0] for patch in patches])
                face_order = np.argsort(face_ids)
                if len(np.unique(face_ids)) != len(face_ids):
                    raise RuntimeError(f"Duplicate ownership in boundary {name!r}")
                boundaries[name] = BoundarySnapshot(
                    *(
                        np.concatenate([patch[i] for patch in patches])[face_order]
                        for i in range(1, 4)
                    )
                )
            snapshot = AnalysisSnapshot(
                solver.step,
                solver.time,
                *fields,
                boundaries,
                solver.max_courant_number,
                max(part[7] for part in parts),
            )
            result = callback(snapshot, *args, **kwargs)
        except BaseException as error:
            failure = error
    solver._collective_io_failure(failure, "application analysis")
    return parallel.bcast(result)


def write_csv(solver, filename, rows, *, columns, append=False):
    """Publish an application table once, relative to the solution directory."""
    import csv
    from pathlib import Path

    failure = None
    if solver.parallel.is_root:
        try:
            path = Path(filename)
            if not path.is_absolute():
                path = Path(solver.solution_dir) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            header = not append or not path.exists() or path.stat().st_size == 0
            with path.open("a" if append else "w", newline="") as stream:
                writer = csv.writer(stream)
                if header:
                    writer.writerow(columns)
                writer.writerows(rows)
        except BaseException as error:
            failure = error
    solver._collective_io_failure(failure, "application CSV output")
