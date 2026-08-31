"""Audit the body-fitted cylinder mesh before any Navier--Stokes run.

The generic curved-surface mesher accepts a broad 35% wall-area tolerance.
This benchmark deliberately tightens that contract to 1% and records the
actual cell count, volumes, solid exclusion, boundary-layer continuity,
wall-normal alignment, and VTK round-trip topology. A mesh that fails here
must never be used for the reference or coupled comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import benchmark_config as cfg  # noqa: E402
from source.solvers.fvm.io.vtk_exporter import write_vtu  # noqa: E402
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry  # noqa: E402
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology  # noqa: E402


def _ideal_surface_distance(points: np.ndarray) -> np.ndarray:
    radial = np.linalg.norm(points[:, :2], axis=1)
    return np.abs(radial - 0.5 * cfg.DIAMETER)


def _wall_block(mesh: dict) -> tuple[np.ndarray, np.ndarray]:
    patch = next(patch for patch in mesh["boundary"] if patch["name"] == "cylinder")
    first = int(patch["start_face"])
    face_ids = np.arange(first, first + int(patch["n_faces"]), dtype=np.int64)
    faces = mesh["faces"]
    return face_ids, np.asarray([np.asarray(faces[index]) for index in face_ids], dtype=object)


def _normal_metrics(face_centres: np.ndarray, area_vectors: np.ndarray) -> dict:
    radial = np.linalg.norm(face_centres[:, :2], axis=1)
    body_outward = np.zeros_like(face_centres)
    body_outward[:, :2] = face_centres[:, :2] / np.maximum(radial[:, None], 1.0e-30)
    cosine = np.einsum("ij,ij->i", area_vectors, body_outward) / np.maximum(
        np.linalg.norm(area_vectors, axis=1), 1.0e-30
    )
    # Boundary normals point out of the fluid domain and therefore into the
    # cylinder: they must oppose the solid's outward normal.
    return {
        "fraction_into_solid": float(np.mean(cosine < 0.0)),
        "median_fluid_vs_body_normal_cosine": float(np.median(cosine)),
        "maximum_fluid_vs_body_normal_cosine": float(np.max(cosine)),
    }


def main() -> None:
    grid = cfg.selected_grid()
    domain_name = cfg.selected_domain_name()
    domain = cfg.selected_reference_domain()
    mesher = cfg.build_mesh(domain, grid)
    print(f"Building cylinder mesh: grid={grid.name}, domain={domain_name}", flush=True)
    mesh = mesher.build()
    topology = validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    face_areas = np.asarray(geometry["face_area"])
    collapsed = np.flatnonzero(face_areas <= 0.0)
    if collapsed.size:
        print(f"Collapsed faces before solver validation: {collapsed.size}")
        n_internal = int(mesh["n_interior_faces"])
        for face_id in collapsed[:12]:
            patch_name = "interior"
            for patch in mesh["boundary"]:
                start = int(patch["start_face"])
                if start <= face_id < start + int(patch["n_faces"]):
                    patch_name = str(patch["name"])
                    break
            vertices = np.asarray(mesh["vertex_position"])[np.asarray(mesh["faces"][face_id])]
            print(
                f"  face={face_id}, patch={patch_name}, owner={mesh['owners'][face_id]}, "
                f"internal={face_id < n_internal}, vertices={vertices.tolist()}"
            )
    orientation = np.einsum(
        "ij,ij->i",
        np.asarray(geometry["face_area_vector"]),
        np.asarray(geometry["cell_connection_vector"]),
    )
    reversed_faces = np.flatnonzero(orientation <= 0.0)
    if reversed_faces.size:
        print(f"Non-positive face orientations before solver validation: {reversed_faces.size}")
        for face_id in reversed_faces[:12]:
            patch_name = "interior"
            for patch in mesh["boundary"]:
                start = int(patch["start_face"])
                if start <= face_id < start + int(patch["n_faces"]):
                    patch_name = str(patch["name"])
                    break
            vertices = np.asarray(mesh["vertex_position"])[np.asarray(mesh["faces"][face_id])]
            print(
                f"  face={face_id}, patch={patch_name}, orientation={orientation[face_id]:.6g}, "
                f"vertices={vertices.tolist()}"
            )
    quality = validate_geometry(mesh, geometry)

    wall_ids, wall_faces = _wall_block(mesh)
    wall_vertices = np.unique(np.concatenate(wall_faces.tolist()))
    wall_points = np.asarray(mesh["vertex_position"])[wall_vertices]
    wall_distance = _ideal_surface_distance(wall_points)
    wall_area = float(np.asarray(geometry["face_area"])[wall_ids].sum())
    triangles = np.asarray(mesher.surface.triangles)
    stl_area = 0.5 * float(
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ).sum()
    )
    ideal_area = float(np.pi * cfg.DIAMETER * cfg.CYLINDER_LENGTH)
    wall_area_error = abs(wall_area - ideal_area) / ideal_area

    centres = np.asarray(geometry["cell_centre"])
    radial = np.linalg.norm(centres[:, :2], axis=1)
    inside = (radial < 0.5 * cfg.DIAMETER - 1.0e-10) & (
        np.abs(centres[:, 2]) < 0.5 * cfg.CYLINDER_LENGTH - 1.0e-10
    )
    normal_metrics = _normal_metrics(
        np.asarray(geometry["face_centre"])[wall_ids],
        np.asarray(geometry["face_area_vector"])[wall_ids],
    )
    cell_levels = np.asarray(mesh["cell_levels"], dtype=np.int32)
    level_values, level_counts = np.unique(cell_levels, return_counts=True)
    wall_owner_cells = np.asarray(mesh["owners"], dtype=np.int64)[wall_ids]
    wall_owner_levels = cell_levels[wall_owner_cells]
    wall_owner_sizes = grid.background / np.power(2.0, wall_owner_levels)
    wall_adjacent = np.zeros(int(mesh["n_cells"]), dtype=np.uint8)
    wall_adjacent[wall_owner_cells] = 1
    layer_meta = mesh["mesh_generation"].get("boundary_layer", {})
    layer_index = np.asarray(mesh.get("boundary_layer_index", []), dtype=np.int16)
    expected_layer_cells = int(layer_meta.get("theta_cells", 0)) * int(
        layer_meta.get("z_cells", 0)
    )
    wall_layers = int(layer_meta.get("wall_layers", 0))
    layer_counts = {
        str(layer): int(np.count_nonzero(layer_index == layer))
        for layer in range(wall_layers)
    }
    cell_vertices = np.asarray(mesh["cell_vertex_indices"], dtype=np.int64)
    wall_cell_points = np.asarray(mesh["vertex_position"])[cell_vertices[wall_owner_cells]]
    wall_cell_radius = np.linalg.norm(wall_cell_points[:, :, :2], axis=2)
    measured_first_height = np.ptp(wall_cell_radius, axis=1)
    wall_face_centres = np.asarray(geometry["face_centre"])[wall_ids]
    wall_area_vectors = np.asarray(geometry["face_area_vector"])[wall_ids]
    wall_to_owner = centres[wall_owner_cells] - wall_face_centres
    wall_alignment = np.einsum("ij,ij->i", -wall_area_vectors, wall_to_owner) / np.maximum(
        np.linalg.norm(wall_area_vectors, axis=1)
        * np.linalg.norm(wall_to_owner, axis=1),
        1.0e-30,
    )
    h = grid.surface
    distance_metrics = {
        "mean": float(np.mean(wall_distance)),
        "p95": float(np.quantile(wall_distance, 0.95)),
        "maximum": float(np.max(wall_distance)),
        "p95_over_surface_h": float(np.quantile(wall_distance, 0.95) / h),
        "maximum_over_surface_h": float(np.max(wall_distance) / h),
        "worst_vertices": [
            {
                "position": wall_points[index].tolist(),
                "distance": float(wall_distance[index]),
                "distance_over_surface_h": float(wall_distance[index] / h),
            }
            for index in np.argsort(wall_distance)[-8:][::-1]
        ],
    }

    violations = []
    if int(mesh["n_cells"]) > grid.target_cells:
        violations.append(
            f"cell count {mesh['n_cells']:,} exceeds grid cap {grid.target_cells:,}"
        )
    if int(mesh["n_cells"]) > 1_000_000:
        violations.append("cell count exceeds the verified one-million-cell solver range")
    if np.any(np.asarray(geometry["cell_volume"]) <= 0.0):
        violations.append("one or more cells have non-positive solver volume")
    if np.any(inside):
        violations.append(f"{int(np.count_nonzero(inside))} fluid cell centres lie in the solid")
    if wall_area_error >= 0.01:
        violations.append(f"wall area error {wall_area_error:.3%} is not below 1%")
    if distance_metrics["p95_over_surface_h"] > 0.35:
        violations.append(
            f"95th-percentile wall displacement is {distance_metrics['p95_over_surface_h']:.3g} h"
        )
    if distance_metrics["maximum_over_surface_h"] > 1.5:
        violations.append(
            f"maximum wall displacement is {distance_metrics['maximum_over_surface_h']:.3g} h"
        )
    if normal_metrics["fraction_into_solid"] < 0.99:
        violations.append(
            f"only {normal_metrics['fraction_into_solid']:.2%} of wall normals point into the solid"
        )
    if not layer_meta:
        violations.append("mesh has no body-fitted boundary-layer metadata")
    if wall_layers < 4:
        violations.append(f"only {wall_layers} wall-normal layers were generated")
    incomplete_layers = {
        layer: count
        for layer, count in layer_counts.items()
        if count != expected_layer_cells
    }
    if incomplete_layers:
        violations.append(f"boundary-layer cell rings are incomplete: {incomplete_layers}")
    requested_first_height = float(layer_meta.get("first_cell_height", np.nan))
    first_height_error = np.abs(measured_first_height - requested_first_height) / max(
        requested_first_height, 1.0e-30
    )
    if float(np.quantile(first_height_error, 0.99)) > 0.05:
        violations.append(
            "99th-percentile first-cell-height error exceeds 5%: "
            f"{np.quantile(first_height_error, 0.99):.3%}"
        )
    if float(np.quantile(wall_alignment, 0.01)) < np.cos(np.deg2rad(15.0)):
        violations.append("more than 1% of first-layer cells deviate over 15 degrees from normal")
    if quality["max_non_orthogonality_deg"] >= 60.0:
        violations.append(
            f"maximum non-orthogonality {quality['max_non_orthogonality_deg']:.3f} deg is not below 60"
        )
    if quality["max_skewness"] >= 0.5:
        violations.append(f"maximum skewness {quality['max_skewness']:.3f} is not below 0.5")
    if quality["max_aspect_ratio"] >= 150.0:
        violations.append(
            f"maximum aspect ratio {quality['max_aspect_ratio']:.3f} is not below 150"
        )

    report = {
        "schema": "openonda-cylinder-mesh-audit/1",
        "passed": not violations,
        "grid": grid.name,
        "domain": domain_name,
        "surface_sha256": mesher.surface.sha256,
        "surface_triangles": len(triangles),
        "topology": topology,
        "quality": quality,
        "cell_count": int(mesh["n_cells"]),
        "target_cell_count": grid.target_cells,
        "positive_minimum_cell_volume": float(np.min(geometry["cell_volume"])),
        "solid_cell_centres": int(np.count_nonzero(inside)),
        "wall_faces": len(wall_ids),
        "wall_vertices": len(wall_vertices),
        "wall_area": wall_area,
        "stl_area": stl_area,
        "ideal_cylinder_area": ideal_area,
        "wall_area_relative_error_vs_solved_span": wall_area_error,
        "complete_stl_area": stl_area,
        "wall_distance_to_ideal_cylinder": distance_metrics,
        "wall_normals": normal_metrics,
        "surface_projection": mesh["mesh_generation"].get("surface_projection", {}),
        "boundary_layer": {
            **layer_meta,
            "expected_cells_per_ring": expected_layer_cells,
            "cell_count_by_wall_layer": layer_counts,
            "first_cell_height_measured": {
                "minimum": float(np.min(measured_first_height)),
                "median": float(np.median(measured_first_height)),
                "maximum": float(np.max(measured_first_height)),
                "p99_relative_error": float(np.quantile(first_height_error, 0.99)),
            },
            "wall_normal_alignment_cosine": {
                "minimum": float(np.min(wall_alignment)),
                "p01": float(np.quantile(wall_alignment, 0.01)),
                "median": float(np.median(wall_alignment)),
            },
        },
        "refinement": {
            "background_cell_size": grid.background,
            "surface_target_cell_size": grid.surface,
            "shear_layer_target_cell_size": grid.shear_layer,
            "near_wake_target_cell_size": grid.near_wake,
            "downstream_wake_target_cell_size": grid.downstream_wake,
            "cell_count_by_level": {
                str(int(level)): int(count)
                for level, count in zip(level_values, level_counts, strict=True)
            },
            "wall_owner_level_minimum": int(np.min(wall_owner_levels)),
            "wall_owner_level_maximum": int(np.max(wall_owner_levels)),
            "wall_owner_cell_size_minimum": float(np.min(wall_owner_sizes)),
            "wall_owner_cell_size_maximum": float(np.max(wall_owner_sizes)),
        },
        "violations": violations,
    }

    output_dir = (
        CASE_DIR
        / "reference_flow"
        / "solution"
        / "mesh_audits"
        / f"{grid.name}_{domain_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mesh_audit.json"
    mesh_path = output_dir / "mesh.vtu"
    write_vtu(
        str(mesh_path),
        mesh,
        {
            "refinementLevel": cell_levels,
            "cellVolume": np.asarray(geometry["cell_volume"], dtype=np.float64),
            "solidCentre": inside.astype(np.uint8),
            "wallAdjacent": wall_adjacent,
            "boundaryLayerIndex": layer_index,
        },
    )
    try:
        import pyvista as pv

        round_trip = pv.read(mesh_path)
        if round_trip.n_cells != int(mesh["n_cells"]):
            violations.append(
                f"VTK round trip returned {round_trip.n_cells:,} cells, expected {mesh['n_cells']:,}"
            )
        if round_trip.n_points != int(mesh["n_points"]):
            violations.append(
                f"VTK round trip returned {round_trip.n_points:,} points, expected {mesh['n_points']:,}"
            )
    except Exception as error:
        violations.append(f"VTK round trip failed: {type(error).__name__}: {error}")
    report["violations"] = violations
    report["passed"] = not violations
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"Audit written to {report_path}")
    if violations:
        raise SystemExit("CUT-CELL MESH AUDIT FAILED")
    print("CUT-CELL MESH AUDIT PASSED")


if __name__ == "__main__":
    main()
