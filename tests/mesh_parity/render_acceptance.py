#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render deterministic visual evidence for a Cartesian-mesher manifest.

The renderer deliberately uses only native ``.npz`` meshes from the
acceptance bundle.  It produces an overview, three orthogonal sections, a wall
view, and a refinement-transition view for every passing case.  The optional
``--mark-inspected`` flag is a deliberate human-review acknowledgement; the
renderer never treats image generation alone as inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh


def _face_centres(mesh: dict[str, Any]) -> np.ndarray:
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    return np.asarray(
        [points[np.asarray(face, dtype=np.int64)].mean(axis=0) for face in mesh["faces"]],
        dtype=np.float64,
    )


def _wall_face_ids(mesh: dict[str, Any]) -> np.ndarray:
    ids: list[int] = []
    for patch in mesh["boundary"]:
        if str(patch.get("type", "patch")).lower() != "wall":
            continue
        start = int(patch["start_face"])
        stop = start + int(patch["n_faces"])
        ids.extend(range(start, stop))
    return np.asarray(ids, dtype=np.int64)


def _transition_cell_ids(mesh: dict[str, Any]) -> np.ndarray:
    levels = np.asarray(mesh.get("cell_levels", ()), dtype=np.int64)
    if levels.shape != (int(mesh["n_cells"]),):
        return np.empty(0, dtype=np.int64)
    owners = np.asarray(mesh["owners"], dtype=np.int64)
    neighbours = np.asarray(mesh["neighbours"], dtype=np.int64)
    internal = int(mesh["n_interior_faces"])
    changed = np.flatnonzero(levels[owners[:internal]] != levels[neighbours])
    if not len(changed):
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate((owners[changed], neighbours[changed])))


def _limits(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1.0e-12)
    return low - 0.04 * span, high + 0.04 * span


def _quality_text(result: dict[str, Any]) -> str:
    quality = result.get("quality", {})
    configuration = result.get("configuration", {})
    return (
        f"requested h: background={configuration.get('max_cell_size')}  "
        f"wall={configuration.get('boundary_cell_size')}  "
        f"closure={quality.get('max_normalized_area_closure_error', float('nan')):.2e}\n"
        f"non-orth max/p99={quality.get('max_non_orthogonality_deg', float('nan')):.1f}/"
        f"{quality.get('p99_non_orthogonality_deg', float('nan')):.1f} deg  "
        f"skew max={quality.get('max_skewness', float('nan')):.2f}  "
        f"cells={result.get('n_cells')}"
    )


def _save_overview(mesh: dict[str, Any], result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    centres = np.asarray(mesh.get("cell_centre", ()), dtype=np.float64)
    if centres.shape != (int(mesh["n_cells"]), 3):
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

        centres = np.asarray(compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"])
    wall = _face_centres(mesh)[_wall_face_ids(mesh)]
    levels = np.asarray(mesh.get("cell_levels", np.zeros(len(centres))), dtype=np.int64)
    low, high = _limits(np.vstack((centres, wall)) if len(wall) else centres)
    fig = plt.figure(figsize=(8.5, 7.0), dpi=120)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(centres[:, 0], centres[:, 1], centres[:, 2], c=levels, s=3, cmap="viridis")
    if len(wall):
        axis.scatter(wall[:, 0], wall[:, 1], wall[:, 2], c="crimson", s=5, alpha=0.85)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"{result['name']} — cells/levels (wall in red)\n{_quality_text(result)}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_sections(mesh: dict[str, Any], result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    centres = np.asarray(mesh.get("cell_centre", ()), dtype=np.float64)
    if centres.shape != (int(mesh["n_cells"]), 3):
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

        centres = np.asarray(compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"])
    levels = np.asarray(mesh.get("cell_levels", np.zeros(len(centres))), dtype=np.int64)
    low, high = _limits(centres)
    spans = high - low
    axes = (
        (0, 1, 2, "x section", "y", "z"),
        (1, 0, 2, "y section", "x", "z"),
        (2, 0, 1, "z section", "x", "y"),
    )
    fig, subplots = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=120)
    for axis, (normal, horizontal, vertical, title, xlabel, ylabel) in zip(
        subplots, axes, strict=True
    ):
        target = float(np.median(centres[:, normal]))
        tolerance = max(
            0.035 * spans[normal],
            0.5 * float(np.min(np.maximum(np.asarray(mesh.get("cell_sizes", (1.0,))), 1e-12))),
        )
        selected = np.flatnonzero(np.abs(centres[:, normal] - target) <= tolerance)
        if not len(selected):
            selected = np.argsort(np.abs(centres[:, normal] - target))[: min(500, len(centres))]
        axis.scatter(
            centres[selected, horizontal],
            centres[selected, vertical],
            c=levels[selected],
            s=5,
            cmap="viridis",
        )
        axis.set_xlim(low[horizontal], high[horizontal])
        axis.set_ylim(low[vertical], high[vertical])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{result['name']} — orthogonal interior sections\n{_quality_text(result)}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_wall(mesh: dict[str, Any], result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    centres = _face_centres(mesh)
    wall_ids = _wall_face_ids(mesh)
    wall = centres[wall_ids]
    fig = plt.figure(figsize=(8.5, 6.5), dpi=120)
    axis = fig.add_subplot(111, projection="3d")
    if len(wall):
        axis.scatter(wall[:, 0], wall[:, 1], wall[:, 2], c="crimson", s=10)
        low, high = _limits(wall)
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(
        f"{result['name']} — wall close-up ({len(wall)} wall faces)\n{_quality_text(result)}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_transitions(mesh: dict[str, Any], result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    centres = np.asarray(mesh.get("cell_centre", ()), dtype=np.float64)
    if centres.shape != (int(mesh["n_cells"]), 3):
        from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

        centres = np.asarray(compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"])
    transition_ids = _transition_cell_ids(mesh)
    levels = np.asarray(mesh.get("cell_levels", np.zeros(len(centres))), dtype=np.int64)
    fig, axis = plt.subplots(figsize=(8.5, 6.5), dpi=120)
    axis.scatter(centres[:, 0], centres[:, 1], c=levels, s=3, cmap="Greys", alpha=0.25)
    if len(transition_ids):
        axis.scatter(
            centres[transition_ids, 0],
            centres[transition_ids, 1],
            c=levels[transition_ids],
            s=12,
            cmap="plasma",
        )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.2)
    axis.set_title(
        f"{result['name']} — coarse/fine transition cells ({len(transition_ids)})\n{_quality_text(result)}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_manifest(manifest_path: Path, *, mark_inspected: bool = False) -> None:
    # Set a writable cache location before importing pyplot in the renderers.
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(manifest_path.parent / ".matplotlib"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest_path.parent
    for result in manifest.get("results", []):
        if result.get("status") != "pass":
            continue
        name, mode = str(result["name"]).split(":", 1)
        mesh = load_native_mesh(output / "meshes" / name / mode / "mesh.npz")
        image_dir = output / "images" / name / mode
        image_dir.mkdir(parents=True, exist_ok=True)
        _save_overview(mesh, result, image_dir / "overview.png")
        _save_sections(mesh, result, image_dir / "sections.png")
        _save_wall(mesh, result, image_dir / "wall.png")
        _save_transitions(mesh, result, image_dir / "transitions.png")
        result["image_paths"] = [
            str(image_dir / filename)
            for filename in ("overview.png", "sections.png", "wall.png", "transitions.png")
        ]
        if mark_inspected:
            result["image_review"] = "inspected"
    if mark_inspected:
        manifest["image_review_policy"] = (
            "release requires inspected images; human review acknowledged"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mark-inspected",
        action="store_true",
        help="record explicit human inspection after rendering",
    )
    args = parser.parse_args()
    render_manifest(args.manifest, mark_inspected=args.mark_inspected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
