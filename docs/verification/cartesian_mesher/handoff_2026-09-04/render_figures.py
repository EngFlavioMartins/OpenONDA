"""Render the durable figures and derived metrics for the 2026-09-04 handoff.

Run from the repository root with the OpenONDA Python environment::

    MPLCONFIGDIR=/private/tmp/openonda-mpl python \
      docs/verification/cartesian_mesher/handoff_2026-09-04/render_figures.py

The script reads only the copied evidence in this handoff directory.  It does
not invoke either mesher.
"""

# ruff: noqa: E402,I001 -- repository imports follow an explicit local path bootstrap.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.mesh_parity.compare_meshes import (
    ComparisonOptions,
    _boundary_face_pairs,
    _build_cell_mapping,
    _mesh_geometry,
    _mesh_topology,
)
from tools.mesh_parity.openfoam_poly_mesh import PolyMesh, read_poly_mesh


HERE = Path(__file__).resolve().parent
BEST = HERE / "evidence" / "best"
FIGURES = HERE / "figures"

CFMESH_COLOUR = "#1f77b4"
OPENONDA_COLOUR = "#ff7f0e"
PASS_COLOUR = "#2ca02c"
FAIL_COLOUR = "#d62728"
GRID_COLOUR = "#d9dee7"


def _report(name: str) -> dict[str, object]:
    return json.loads((BEST / name / "parity_report.json").read_text(encoding="utf-8"))


def _meshes(name: str) -> tuple[PolyMesh, PolyMesh]:
    base = BEST / name
    return (
        read_poly_mesh(base / "cfmesh" / "constant" / "polyMesh"),
        read_poly_mesh(base / "openonda" / "constant" / "polyMesh"),
    )


def _body_points(mesh: PolyMesh) -> np.ndarray:
    patch = next(item for item in mesh.boundary if item.name == "body")
    face_ids = range(patch.start_face, patch.start_face + patch.n_faces)
    point_ids = np.unique(np.concatenate([mesh.faces[face_id] for face_id in face_ids]))
    return mesh.points[point_ids]


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID_COLOUR, linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def render_status_overview() -> None:
    aligned = _report("aligned_final")
    oblique = _report("oblique_final")
    cylinder = _report("cylinder_final")
    stages = [
        ("surface projection", _report("cylinder_projection")),
        ("patch assignment", _report("cylinder_patch")),
        ("edge extraction", _report("cylinder_edge")),
        ("boundary wrapper", _report("cylinder_wrapper")),
        ("final optimisation", cylinder),
    ]

    fig = plt.figure(figsize=(13.6, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.72, 1.28), hspace=0.34, wspace=0.25)
    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.98,
        "OpenONDA Cartesian mesher — cfMesh parity checkpoint",
        fontsize=21,
        fontweight="bold",
        va="top",
    )
    title_axis.text(
        0.0,
        0.70,
        "Clean native oracle • OpenFOAM-v2412 • OMP_NUM_THREADS=1 • 2026-09-04",
        fontsize=11.5,
        color="#4b5563",
        va="top",
    )

    aligned_cells = aligned["comparison"]["cell_mapping"]["matched_cells"]
    oblique_geometry = oblique["comparison"]["geometry"]
    oblique_angle = oblique_geometry["boundary_face_normal_angle_degrees"]["max"]
    oblique_gate = oblique["comparison_options"]["face_normal_angle_tolerance_degrees"]
    cylinder_mapping = cylinder["comparison"]["cell_mapping"]
    cards = [
        (
            "ALIGNED CUBE",
            PASS_COLOUR,
            "PASS",
            f"{aligned_cells:,}/{aligned_cells:,} cells mapped\nexact topology + geometry",
        ),
        (
            "OBLIQUE CUBE",
            FAIL_COLOUR,
            "NEAR MISS",
            f"exact topology; all cells mapped\nnormal {oblique_angle:.7f}° > {oblique_gate:.3f}°",
        ),
        (
            "CURVED CYLINDER",
            FAIL_COLOUR,
            "OPEN",
            f"exact global invariants\n{cylinder_mapping['matched_cells']:,}/{cylinder['comparison']['cfmesh']['n_cells']:,} cells tightly mapped",
        ),
    ]
    for index, (label, colour, status, detail) in enumerate(cards):
        left = index / 3.0
        title_axis.add_patch(
            plt.Rectangle(
                (left + 0.007, 0.02),
                0.307,
                0.45,
                transform=title_axis.transAxes,
                facecolor="#f7f8fa",
                edgecolor=GRID_COLOUR,
                linewidth=1.0,
            )
        )
        title_axis.add_patch(
            plt.Rectangle(
                (left + 0.007, 0.02),
                0.009,
                0.45,
                transform=title_axis.transAxes,
                facecolor=colour,
                edgecolor=colour,
            )
        )
        title_axis.text(left + 0.033, 0.39, label, fontsize=10, color="#4b5563", va="top")
        title_axis.text(
            left + 0.033, 0.29, status, fontsize=16, fontweight="bold", color=colour, va="top"
        )
        title_axis.text(left + 0.033, 0.17, detail, fontsize=10.5, color="#111827", va="top")

    stage_axis = fig.add_subplot(grid[1, 0])
    labels = [label for label, _ in stages]
    passed = [report["status"] == "pass" for _, report in stages]
    y_values = np.arange(len(stages))
    stage_axis.barh(
        y_values,
        np.ones(len(stages)),
        color=[PASS_COLOUR if item else FAIL_COLOUR for item in passed],
        height=0.62,
    )
    stage_axis.set_yticks(y_values, labels)
    stage_axis.set_xlim(0.0, 1.0)
    stage_axis.set_xticks([])
    stage_axis.invert_yaxis()
    stage_axis.set_title("Curved-cylinder checkpoint ladder", loc="left", fontweight="bold")
    for y_value, is_pass in zip(y_values, passed, strict=True):
        stage_axis.text(
            0.97,
            y_value,
            "PASS" if is_pass else "OPEN",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
        )
    stage_axis.spines[:].set_visible(False)

    gate_axis = fig.add_subplot(grid[1, 1])
    gate_labels = [
        "Aligned normal angle",
        "Oblique normal angle",
        "Oblique volume error",
        "Cylinder mapped cells",
    ]
    aligned_angle = aligned["comparison"]["geometry"]["boundary_face_normal_angle_degrees"]["max"]
    aligned_gate = aligned["comparison_options"]["face_normal_angle_tolerance_degrees"]
    oblique_volume = oblique_geometry["cell_volume_relative_error"]["max"]
    oblique_volume_gate = oblique["comparison_options"]["volume_relative_tolerance"]
    cylinder_ratio = cylinder_mapping["matched_cells"] / cylinder["comparison"]["cfmesh"]["n_cells"]
    gate_ratios = [
        aligned_angle / aligned_gate,
        oblique_angle / oblique_gate,
        oblique_volume / oblique_volume_gate,
        cylinder_ratio,
    ]
    colours = [PASS_COLOUR, FAIL_COLOUR, PASS_COLOUR, FAIL_COLOUR]
    gate_axis.barh(np.arange(4), gate_ratios, color=colours, height=0.62)
    gate_axis.axvline(1.0, color="#111827", linestyle="--", linewidth=1.1, label="gate / full map")
    gate_axis.set_yticks(np.arange(4), gate_labels)
    gate_axis.invert_yaxis()
    gate_axis.set_xlim(0.0, 1.12)
    gate_axis.set_xlabel("ratio to gate (lower is better; mapped-cells target is 1.0)")
    gate_axis.set_title("Strict final-gate position", loc="left", fontweight="bold")
    gate_axis.legend(frameon=False, loc="lower right")
    _style_axis(gate_axis)

    fig.savefig(FIGURES / "01_parity_status_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_boundary_overlays() -> dict[str, object]:
    cases = (("aligned_final", "Aligned cube"), ("oblique_final", "Oblique cube"))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), facecolor="white")
    derived: dict[str, object] = {}
    for row, (name, title) in enumerate(cases):
        cfmesh, openonda = _meshes(name)
        cf_points = _body_points(cfmesh)
        oo_points = _body_points(openonda)
        derived[name] = {
            "cfmesh_body_points": int(len(cf_points)),
            "openonda_body_points": int(len(oo_points)),
        }
        for column, coordinates in enumerate(((0, 1), (0, 2))):
            axis = axes[row, column]
            axis.scatter(
                cf_points[:, coordinates[0]],
                cf_points[:, coordinates[1]],
                s=18,
                facecolors="none",
                edgecolors=CFMESH_COLOUR,
                linewidths=0.8,
                label="cfMesh",
            )
            axis.scatter(
                oo_points[:, coordinates[0]],
                oo_points[:, coordinates[1]],
                s=7,
                color=OPENONDA_COLOUR,
                alpha=0.72,
                label="OpenONDA",
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y" if column == 0 else "z")
            axis.set_title(f"{title}: {'x–y' if column == 0 else 'x–z'} boundary-point overlay")
            _style_axis(axis)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle(
        "Body-boundary overlays (cfMesh rings; OpenONDA dots)",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "02_cube_boundary_overlays.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return derived


def _options(report: dict[str, object]) -> ComparisonOptions:
    return ComparisonOptions(**report["comparison_options"])


def render_oblique_normal_error() -> dict[str, object]:
    report = _report("oblique_final")
    cfmesh, openonda = _meshes("oblique_final")
    patch_names = tuple(sorted(patch.name for patch in cfmesh.boundary))
    cf_topology = _mesh_topology(cfmesh, patch_names)
    oo_topology = _mesh_topology(openonda, patch_names)
    cf_geometry = _mesh_geometry(cfmesh)
    oo_geometry = _mesh_geometry(openonda)
    mapping, mapping_report = _build_cell_mapping(
        cf_geometry, oo_geometry, cf_topology, oo_topology, _options(report)
    )
    if not mapping_report["complete"]:
        raise RuntimeError("The saved oblique case must have a complete cell mapping")
    face_pairs = _boundary_face_pairs(cfmesh, openonda, cf_topology, oo_topology, mapping)
    cf_faces = np.asarray([pair[0] for pair in face_pairs], dtype=np.int64)
    oo_faces = np.asarray([pair[1] for pair in face_pairs], dtype=np.int64)
    cf_normals = cf_geometry.face_area_vectors[cf_faces]
    oo_normals = oo_geometry.face_area_vectors[oo_faces]
    dot = np.einsum("ij,ij->i", cf_normals, oo_normals)
    dot /= np.linalg.norm(cf_normals, axis=1) * np.linalg.norm(oo_normals, axis=1)
    angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    gate = float(report["comparison_options"]["face_normal_angle_tolerance_degrees"])
    maximum_index = int(np.argmax(angles))
    source_face = int(cf_faces[maximum_index])
    target_face = int(oo_faces[maximum_index])
    patch_id = int(cf_topology.face_patch_ids[source_face])
    centre = cf_geometry.face_centres[source_face]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), facecolor="white")
    sorted_angles = np.sort(angles)
    rank = 100.0 * (np.arange(len(sorted_angles)) + 1) / len(sorted_angles)
    axes[0].semilogy(rank, np.maximum(sorted_angles, 1.0e-12), color=CFMESH_COLOUR, linewidth=1.8)
    axes[0].axhline(
        gate, color=FAIL_COLOUR, linestyle="--", linewidth=1.2, label=f"gate {gate:.3f}°"
    )
    axes[0].scatter([100.0], [angles[maximum_index]], color=FAIL_COLOUR, s=42, zorder=3)
    axes[0].set_xlabel("boundary-face percentile")
    axes[0].set_ylabel("normal-angle difference (degrees, log scale)")
    axes[0].set_title("All 1,066 paired boundary faces")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    plot = axes[1].scatter(
        cf_geometry.face_centres[cf_faces, 0],
        cf_geometry.face_centres[cf_faces, 1],
        c=angles,
        s=18,
        cmap="magma",
        norm="log",
        vmin=max(float(np.min(angles[angles > 0.0])), 1.0e-8),
        vmax=float(np.max(angles)),
    )
    axes[1].scatter(
        [centre[0]], [centre[1]], facecolors="none", edgecolors="#00d5ff", s=130, linewidths=2.0
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"Face locations; max at ({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f})")
    _style_axis(axes[1])
    colour_bar = fig.colorbar(plot, ax=axes[1], pad=0.02)
    colour_bar.set_label("normal-angle difference (degrees)")

    fig.suptitle(
        f"Oblique cube: exact topology, one strict geometry near-miss (max {angles[maximum_index]:.7f}°)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "03_oblique_normal_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "paired_boundary_faces": int(len(face_pairs)),
        "gate_degrees": gate,
        "max_angle_degrees": float(angles[maximum_index]),
        "faces_above_gate": int(np.count_nonzero(angles > gate)),
        "cfmesh_face_id": source_face,
        "openonda_face_id": target_face,
        "patch": patch_names[patch_id],
        "cfmesh_face_centre": centre.tolist(),
    }


def render_cylinder_diagnostics() -> dict[str, object]:
    report = _report("cylinder_final")
    cfmesh, openonda = _meshes("cylinder_final")
    cf_geometry = _mesh_geometry(cfmesh)
    oo_geometry = _mesh_geometry(openonda)
    distances, nearest = cKDTree(oo_geometry.cell_centres).query(cf_geometry.cell_centres, k=1)
    tolerance = float(report["comparison"]["cell_mapping"]["candidate_centroid_tolerance"])
    farthest = int(np.argmax(distances))
    cf_body = _body_points(cfmesh)
    oo_body = _body_points(openonda)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), facecolor="white")
    axes[0].scatter(
        cf_body[:, 0],
        cf_body[:, 1],
        s=23,
        facecolors="none",
        edgecolors=CFMESH_COLOUR,
        linewidths=0.8,
        label="cfMesh",
    )
    axes[0].scatter(
        oo_body[:, 0],
        oo_body[:, 1],
        s=8,
        color=OPENONDA_COLOUR,
        alpha=0.65,
        label="OpenONDA",
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Body boundary points, end-on overlay")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    colours = np.where(distances <= tolerance, PASS_COLOUR, FAIL_COLOUR)
    axes[1].scatter(
        cf_geometry.cell_centres[:, 2],
        np.maximum(distances, 1.0e-12),
        c=colours,
        s=9,
        alpha=0.66,
        linewidths=0.0,
    )
    axes[1].axhline(
        tolerance,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"tight centroid radius {tolerance:.3e}",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cfMesh cell-centre z")
    axes[1].set_ylabel("nearest OpenONDA cell-centre distance")
    axes[1].set_title("Final displacement is distributed along the cylinder")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])

    fig.suptitle(
        "Curved cylinder: exact mesh counts and histograms, divergent final coordinates",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "04_cylinder_mapping_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "cells": int(cfmesh.n_cells),
        "reported_tight_matches": int(report["comparison"]["cell_mapping"]["matched_cells"]),
        "reported_tight_unmatched": int(
            report["comparison"]["cell_mapping"]["unmatched_cfmesh_cells"]
        ),
        "candidate_centroid_tolerance": tolerance,
        "nearest_neighbour_within_tolerance": int(np.count_nonzero(distances <= tolerance)),
        "nearest_neighbour_distance": {
            "max": float(np.max(distances)),
            "mean": float(np.mean(distances)),
            "p95": float(np.percentile(distances, 95.0)),
            "p99": float(np.percentile(distances, 99.0)),
        },
        "farthest_cfmesh_cell": farthest,
        "farthest_cfmesh_cell_centre": cf_geometry.cell_centres[farthest].tolist(),
        "nearest_openonda_cell": int(nearest[farthest]),
        "nearest_openonda_cell_centre": oo_geometry.cell_centres[int(nearest[farthest])].tolist(),
        "cfmesh_body_points": int(len(cf_body)),
        "openonda_body_points": int(len(oo_body)),
    }


def write_manifest() -> None:
    entries: list[str] = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(HERE)}")
    (HERE / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="ascii")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_status_overview()
    derived = {
        "cube_boundary_overlays": render_boundary_overlays(),
        "oblique_normal_error": render_oblique_normal_error(),
        "cylinder_diagnostics": render_cylinder_diagnostics(),
    }
    (HERE / "derived_metrics.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest()


if __name__ == "__main__":
    main()
