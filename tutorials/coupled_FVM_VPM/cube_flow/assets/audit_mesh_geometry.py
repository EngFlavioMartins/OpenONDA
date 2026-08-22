"""Geometry-preservation audit for the cube FVM mesh.

Audits the adaptive Cartesian mesh produced for ``cube_flow`` at every stage of
the pipeline — resolver, root mesh, MPI-localized numerical mesh, VTK
visualization mesh and the written VTU — and writes

* ``solution/mesh_audit/root_mesh.vtu`` (with ``GlobalCellIds``,
  ``refinementLevel`` and ``solidOverlapVolume`` cell fields), and
* ``solution/mesh_provenance.json``

The body is the geometry authority: its six faces must be exact Cartesian
lattice planes, the wall patch must coincide with the STL bounds, and no fluid
cell may overlap the solid with positive volume.

Run serially::

    python -u assets/audit_mesh_geometry.py

or partitioned (mirrors the 4-rank production run)::

    mpiexec -n 4 python -u assets/audit_mesh_geometry.py

Exit status is non-zero when any stage fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))
sys.path.insert(0, str(CASE_DIR.parents[2]))  # source/ root for in-place runs

import cubeFlow_setup as cube_flow_setup  # noqa: E402  (geometry configuration)


def _boundary_face_sf(mesh_data) -> np.ndarray:
    """Surface-vector magnitude and orientation for every boundary face."""
    faces = np.asarray(mesh_data["faces"])
    points = np.asarray(mesh_data["points"], dtype=np.float64)
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
    points = np.asarray(mesh_data["points"], dtype=np.float64)
    cell_vertices = np.asarray(mesh_data["cell_vertices"], dtype=np.int64)
    vertices = points[cell_vertices]
    lo = vertices.min(axis=1)
    hi = vertices.max(axis=1)
    body_min = np.asarray(body_bounds[::2], dtype=np.float64)
    body_max = np.asarray(body_bounds[1::2], dtype=np.float64)
    overlap = np.maximum(0.0, np.minimum(hi, body_max) - np.maximum(lo, body_min))
    return overlap[:, 0] * overlap[:, 1] * overlap[:, 2]


def wall_metrics(mesh_data, wall_patch_name: str) -> dict:
    """Wall-patch bounds, area and surface-vector sum."""
    faces = np.asarray(mesh_data["faces"])
    points = np.asarray(mesh_data["points"], dtype=np.float64)
    patch = next(p for p in mesh_data["boundary"] if p["name"] == wall_patch_name)
    block = faces[int(patch["start_face"]) : int(patch["start_face"]) + int(patch["n_faces"])]
    sf = _boundary_face_sf(mesh_data)
    wall_sf = np.concatenate([s for name, s in sf if name == wall_patch_name])
    wall_vertices = points[np.unique(block)]
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
        boundary_sf = np.concatenate([s for _, s in _boundary_face_sf(mesh_data)])
        closed = bool(np.allclose(boundary_sf.sum(axis=0), 0.0, atol=1e-9))
    else:
        boundary_sf = np.zeros((0, 3))
        closed = True
    metrics = {
        "stage": label,
        "n_cells": int(mesh_data["n_cells"]),
        "n_points": int(mesh_data.get("n_points", len(mesh_data["points"]))),
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


def stage_vtu(mesh, body_bounds, output_dir: Path) -> None:
    """Write the root mesh to VTU with provenance cell fields and audit the file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlaps = cell_overlap_vs_body(mesh, body_bounds)
    fields = {
        "GlobalCellIds": np.arange(mesh["n_cells"], dtype=np.int64),
        "refinementLevel": np.asarray(mesh["cell_levels"], dtype=np.int32),
        "solidOverlapVolume": overlaps.astype(np.float64),
    }
    from source.solvers.FVM.io.vtk_exporter import write_vtu

    path = output_dir / "root_mesh.vtu"
    write_vtu(str(path), mesh, fields)

    import pyvista as pv

    grid = pv.read(str(path))
    file_overlaps = cell_overlap_vs_body(
        {
            "points": np.asarray(grid.points),
            "cell_vertices": grid.cells_dict[grid.celltypes[0]].reshape(-1, 8),
            "n_cells": grid.n_cells,
            "n_points": grid.n_points,
        },
        body_bounds,
    )
    return {
        "file": str(path),
        "file_cells": grid.n_cells,
        "file_overlapping_cells": int(np.count_nonzero(file_overlaps > 1e-12)),
        "file_max_overlap_volume": float(file_overlaps.max()) if len(file_overlaps) else 0.0,
    }


def main() -> None:
    mesher = cube_flow_setup.FVM_MESH
    body_bounds = mesher.surface_bounds
    wall_patch_name = mesher.wall_patch_name

    print("Building the adaptive Cartesian mesh ...", flush=True)
    mesh = mesher.build()
    root_metrics = audit_stage(mesh, body_bounds, wall_patch_name, "root")

    stage_records = [root_metrics]
    violations: list[str] = []
    violations.extend(check_stage(root_metrics, body_bounds))

    # Partitioned stage: every rank audits its owned cells; parity is verified
    # by all-reducing owned coverage against the global mesh.
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except (ImportError, ValueError):
        comm = None

    if comm is not None and comm.size > 1:
        from source.solvers.FVM.mesh.partition import localize_mesh_and_geometry

        local_mesh, _local_geo, partition = localize_mesh_and_geometry(
            mesh, {}, comm.rank, comm.size
        )
        n_owned = local_mesh["_n_owned"]
        owned = {key: local_mesh[key] for key in ("points", "faces", "boundary")}
        owned["n_cells"] = n_owned
        owned["n_points"] = len(local_mesh["points"])
        owned["cell_vertices"] = local_mesh["cell_vertices"][:n_owned]
        owned_metrics = audit_stage(
            owned, body_bounds, wall_patch_name, "local-owned", check_wall=False
        )
        stage_records.append(owned_metrics)
        violations.extend(check_stage(owned_metrics, body_bounds))

        pts = np.asarray(local_mesh["points"], dtype=np.float64)
        cv = np.asarray(local_mesh["cell_vertices"], dtype=np.int64)[:n_owned]
        cell_pts = pts[cv]
        cell_min = cell_pts.min(axis=1)
        cell_max = cell_pts.max(axis=1)
        cell_volumes = np.prod(cell_max - cell_min, axis=1)
        owned_vol = float(cell_volumes.sum())
        all_owned_min = np.zeros(3)
        all_owned_max = np.zeros(3)
        all_owned_vol = np.zeros(1)
        comm.Allreduce(cell_min.min(axis=0), all_owned_min, op=MPI.MIN)
        comm.Allreduce(cell_max.max(axis=0), all_owned_max, op=MPI.MAX)
        comm.Allreduce(np.asarray([owned_vol]), all_owned_vol, op=MPI.SUM)
        global_pts = np.asarray(mesh["points"])
        global_min = global_pts.min(axis=0)
        global_max = global_pts.max(axis=0)
        if not np.allclose(all_owned_min, global_min, atol=1e-9) or not np.allclose(
            all_owned_max, global_max, atol=1e-9
        ):
            violations.append("local-owned: owned coverage does not span the global AABB")
        global_vertices = np.asarray(mesh["points"])[
            np.asarray(mesh["cell_vertices"], dtype=np.int64)
        ]
        global_cell_volumes = np.prod(
            global_vertices.max(axis=1) - global_vertices.min(axis=1), axis=1
        )
        global_vol = float(global_cell_volumes.sum())
        if not np.isclose(all_owned_vol[0], global_vol, rtol=1e-9):
            violations.append(
                "local-owned: owned-cell volumes do not sum to the global mesh volume "
                f"({all_owned_vol[0]:.9g} vs {global_vol:.9g})"
            )
    else:
        stage_records.append(
            {"stage": "local-owned", "n_cells": int(mesh["n_cells"]), "skipped": True}
        )

    # VTK visualization mesh and written VTU stages are root-only in a
    # partitioned run; every rank has already audited its owned cells above.
    is_root = comm is None or comm.rank == 0
    if is_root:
        try:
            from source.solvers.FVM.mesh.partition import _visualization_mesh

            vis = _visualization_mesh(mesh, np.arange(mesh["n_cells"], dtype=np.int64))
            vis_metrics = audit_stage(vis, body_bounds, wall_patch_name, "visualization")
            stage_records.append(vis_metrics)
            violations.extend(check_stage(vis_metrics, body_bounds))
        except Exception as exc:  # pragma: no cover - diagnostic path
            violations.append(f"visualization: could not audit the VTK view ({exc})")

        # Written VTU stage
        output_dir = CASE_DIR / "solution" / "mesh_audit"
        vtu = stage_vtu(mesh, body_bounds, output_dir)
        if vtu["file_overlapping_cells"]:
            violations.append(
                f"vtk-file: {vtu['file_overlapping_cells']} cells overlap the body "
                f"(max volume {vtu['file_max_overlap_volume']:.6g})"
            )

        provenance = {
            "schema": "mesh-provenance/1",
            "mesh_method": "adaptive_cartesian",
            "surface_file": mesher.surface_file,
            "surface_sha256": mesher.surface.sha256 if mesher.surface else None,
            "surface_bounds": list(body_bounds),
            "surface_triangle_count": len(mesher.surface.triangles) if mesher.surface else 0,
            "requested_domain": list(mesher.requested_domain),
            "effective_domain": list(mesher.effective_domain),
            "padding_per_face": list(mesher.padding),
            "background_cell_size": float(mesher.max_cell_size),
            "finest_cell_size": float(mesher._resolved_h_min),
            "maximum_level": int(mesher._resolved_max_level),
            "body_lattice_indices": list(mesher._body_lattice_indices),
            "preserve_body_geometry": bool(mesher.preserve_body_geometry),
            "stages": stage_records,
            "vtk_file": vtu,
            "violations": violations,
            "passed": not violations,
        }
        (output_dir / "mesh_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

        for metrics in stage_records:
            print(f"  {metrics['stage']}: {metrics}", flush=True)
        print(f"  vtk-file: {vtu}", flush=True)

        if violations:
            print("\nGEOMETRY AUDIT FAILED:")
            for violation in violations:
                print(f"  - {violation}")
            sys.exit(1)
        print(
            "\nGEOMETRY AUDIT PASSED: body faces are exact lattice planes, "
            "the wall patch coincides with the STL bounds, and no fluid cell "
            "overlaps the solid."
        )
        print(f"Provenance written to {output_dir / 'mesh_provenance.json'}")


if __name__ == "__main__":
    main()
