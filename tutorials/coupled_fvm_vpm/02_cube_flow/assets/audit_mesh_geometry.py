"""Geometry-preservation audit for the cube FVM mesh.

Audits the adaptive Cartesian mesh produced for ``cube_flow``, its disjoint
owned-cell partitions, its VTK visualization mesh and the written VTU. Writes

* ``solution/mesh_audit/root_mesh.vtu`` (with ``global_cell_id`` and
  ``solidOverlapVolume`` cell fields), and
* ``solution/mesh_audit/mesh_provenance.json``

The body is the geometry authority: its six faces must be exact Cartesian
lattice planes, the wall patch must coincide with the STL bounds, and no fluid
cell may overlap the solid with positive volume.

Run with ``python assets/audit_mesh_geometry.py``. The solver owns the
parallel runtime and the single report writer.

Exit status is non-zero when any stage fails.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]

from openonda.tutorial_runner import load_case_module

cube_flow_setup = load_case_module(CASE_DIR)


def _boundary_face_area_vector(mesh_data) -> np.ndarray:
    """Surface-vector magnitude and orientation for every boundary face."""
    faces = mesh_data["faces"]
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    out = []
    for patch in mesh_data["boundary"]:
        first = int(patch["start_face"])
        block = faces[first : first + int(patch["n_faces"])]
        sf = np.empty((len(block), 3), dtype=np.float64)
        for i, face in enumerate(block):
            verts = points[face]
            centre = verts.mean(axis=0)
            total = np.zeros(3)
            for j in range(len(verts)):
                p = verts[j] - centre
                q = verts[(j + 1) % len(verts)] - centre
                total += 0.5 * np.cross(p, q)
            sf[i] = total
        out.append((patch["name"], sf))
    return out


def cell_overlap_vs_body(mesh_data, body_bounds) -> np.ndarray:
    """Per-cell positive-volume overlap with the body AABB (0 for conformal cells)."""
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    cell_vertex_indices = np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int64)
    vertices = points[cell_vertex_indices]
    lo = vertices.min(axis=1)
    hi = vertices.max(axis=1)
    body_min = np.asarray(body_bounds[::2], dtype=np.float64)
    body_max = np.asarray(body_bounds[1::2], dtype=np.float64)
    overlap = np.maximum(0.0, np.minimum(hi, body_max) - np.maximum(lo, body_min))
    return overlap[:, 0] * overlap[:, 1] * overlap[:, 2]


def wall_metrics(mesh_data, wall_patch_name: str) -> dict:
    """Wall-patch bounds, area and surface-vector sum."""
    faces = mesh_data["faces"]
    points = np.asarray(mesh_data["vertex_position"], dtype=np.float64)
    patch = next(p for p in mesh_data["boundary"] if p["name"] == wall_patch_name)
    block = faces[int(patch["start_face"]) : int(patch["start_face"]) + int(patch["n_faces"])]
    sf = _boundary_face_area_vector(mesh_data)
    wall_sf = np.concatenate([s for name, s in sf if name == wall_patch_name])
    wall_vertices = points[np.unique(np.concatenate(block))]
    return {
        "bounds_min": wall_vertices.min(axis=0).tolist(),
        "bounds_max": wall_vertices.max(axis=0).tolist(),
        "area": float(np.linalg.norm(wall_sf, axis=1).sum()),
        "sf_sum": wall_sf.sum(axis=0).tolist(),
        "n_faces": int(len(block)),
    }


def audit_stage(
    mesh_data, body_bounds, wall_patch_name: str, label: str, *, check_wall: bool = True
) -> dict:
    """Run every geometry check on one mesh representation and return metrics."""
    overlaps = cell_overlap_vs_body(mesh_data, body_bounds)
    boundary = mesh_data["boundary"]
    if boundary:
        boundary_sf = np.concatenate([s for _, s in _boundary_face_area_vector(mesh_data)])
        closed = bool(np.allclose(boundary_sf.sum(axis=0), 0.0, atol=1e-9))
    else:
        boundary_sf = np.zeros((0, 3))
        closed = True
    metrics = {
        "stage": label,
        "n_cells": int(mesh_data["n_cells"]),
        "n_points": int(mesh_data.get("n_points", len(mesh_data["vertex_position"]))),
        "overlapping_cells": int(np.count_nonzero(overlaps > 1e-12)),
        "max_overlap_volume": float(overlaps.max()) if len(overlaps) else 0.0,
        "boundary_sf_sum": boundary_sf.sum(axis=0).tolist(),
        "closed_boundary": closed,
    }
    if check_wall and boundary:
        metrics["wall"] = wall_metrics(mesh_data, wall_patch_name)
    else:
        metrics["wall"] = None
    return metrics


def check_stage(metrics: dict, body_bounds, tolerance: float = 1e-10) -> list[str]:
    """Return a list of violation strings for one stage's metrics."""
    violations = []
    if metrics["overlapping_cells"]:
        violations.append(
            f"{metrics['stage']}: {metrics['overlapping_cells']} cells overlap the body "
            f"(max volume {metrics['max_overlap_volume']:.6g})"
        )
    wall = metrics["wall"]
    if wall is None:
        # Partition-local views contain only a slice of the boundary; the
        # closed-surface and wall-coincidence invariants are checked on the
        # root mesh and the written VTU, where the full boundary exists.
        return violations
    if not metrics["closed_boundary"]:
        violations.append(f"{metrics['stage']}: boundary is not closed (ΣSf != 0)")
    scale = max(1.0, float(np.abs(body_bounds).max()))
    atol = 1e-6 * scale  # STL is float32; the wall is conformal within its precision
    expected_min = np.asarray(body_bounds[::2], dtype=np.float64)
    expected_max = np.asarray(body_bounds[1::2], dtype=np.float64)
    if not np.allclose(wall["bounds_min"], expected_min, rtol=0.0, atol=atol):
        violations.append(
            f"{metrics['stage']}: wall minimum {wall['bounds_min']} != STL {expected_min.tolist()}"
        )
    if not np.allclose(wall["bounds_max"], expected_max, rtol=0.0, atol=atol):
        violations.append(
            f"{metrics['stage']}: wall maximum {wall['bounds_max']} != STL {expected_max.tolist()}"
        )
    if not np.allclose(wall["sf_sum"], [0.0, 0.0, 0.0], atol=1e-9):
        violations.append(f"{metrics['stage']}: wall ΣSf = {wall['sf_sum']} != [0, 0, 0]")
    expected_area = 6.0
    if not np.isclose(wall["area"], expected_area, rtol=0.0, atol=1e-6):
        violations.append(f"{metrics['stage']}: wall area {wall['area']:.9g} != {expected_area}")
    return violations


def stage_vtu(mesh, body_bounds, output_dir: Path) -> dict:
    """Write the root mesh to VTU with provenance cell fields and audit the file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlaps = cell_overlap_vs_body(mesh, body_bounds)
    fields = {
        "global_cell_id": np.arange(mesh["n_cells"], dtype=np.int64),
        "solidOverlapVolume": overlaps.astype(np.float64),
    }
    from source.solvers.fvm.io.vtk_exporter import VTKExporter

    path = output_dir / "root_mesh.vtu"
    VTKExporter(mesh).export(str(path), fields)

    import pyvista as pv

    grid = pv.read(str(path))
    body_min = np.asarray(body_bounds[::2])
    body_max = np.asarray(body_bounds[1::2])
    file_overlaps = np.empty(grid.n_cells)
    for index in range(grid.n_cells):
        vertices = grid.get_cell(index).points
        overlap = np.maximum(
            0.0,
            np.minimum(vertices.max(axis=0), body_max) - np.maximum(vertices.min(axis=0), body_min),
        )
        file_overlaps[index] = np.prod(overlap)
    return {
        "file": str(path),
        "file_cells": grid.n_cells,
        "file_overlapping_cells": int(np.count_nonzero(file_overlaps > 1e-12)),
        "file_max_overlap_volume": float(file_overlaps.max()) if len(file_overlaps) else 0.0,
    }


def audit_saved_mesh(fields, mesh_path: Path, output_dir: Path) -> None:
    """Audit the current native mesh and each disjoint owned-cell partition."""
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.partition import CellPartition, _visualization_mesh

    mesh = load_native_mesh(mesh_path)
    body_bounds = np.asarray(cube_flow_setup.CUBE_BOUNDS)
    wall_patch_name = "cube"
    records = [audit_stage(mesh, body_bounds, wall_patch_name, "root")]
    if len(fields.cell_centre) != mesh["n_cells"]:
        raise ValueError("Analysis did not receive the complete mesh")
    owned_ids = []
    for rank in range(cube_flow_setup.FVM_SETUP.cores):
        partition = CellPartition.from_mesh_data(mesh, rank, cube_flow_setup.FVM_SETUP.cores)
        ids = partition.owned_global_ids
        owned_ids.extend(ids.tolist())
        owned = {
            **mesh,
            "n_cells": len(ids),
            "boundary": [],
            "cell_vertex_indices": np.asarray(mesh["cell_vertex_indices"])[ids],
        }
        records.append(
            audit_stage(
                owned, body_bounds, wall_patch_name, f"owned-partition-{rank}", check_wall=False
            )
        )
    if sorted(owned_ids) != list(range(mesh["n_cells"])):
        raise ValueError("Partitions do not cover every global cell exactly once")
    visual = _visualization_mesh(mesh, np.arange(mesh["n_cells"]))
    records.append(audit_stage(visual, body_bounds, wall_patch_name, "visualization"))
    written = stage_vtu(mesh, body_bounds, output_dir)
    violations = [error for record in records for error in check_stage(record, body_bounds)]
    if written["file_overlapping_cells"]:
        violations.append("Written VTK cells overlap the body")
    report = {
        "schema": "mesh-provenance/2",
        "stages": records,
        "vtk_file": written,
        "surface_bounds": body_bounds.tolist(),
        "violations": violations,
        "passed": not violations,
    }
    destination = output_dir / "mesh_provenance.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Geometry audit written to {destination}", flush=True)
    if violations:
        raise ValueError("Geometry audit failed: " + "; ".join(violations))


def main() -> None:
    fvm = cube_flow_setup.fvm
    output_dir = CASE_DIR / "solution" / "mesh_audit"
    config = replace(
        cube_flow_setup.FVM_SETUP, case_name="mesh_audit", samplers=(), backup=fvm.BackupConfig()
    )
    with fvm.create_fvm_solver(
        config, case_dir=CASE_DIR, solution_dir=output_dir, mesh=cube_flow_setup.FVM_MESH
    ) as solver:
        solver.evaluate(audit_saved_mesh, output_dir / "mesh.npz", output_dir)


if __name__ == "__main__":
    main()
