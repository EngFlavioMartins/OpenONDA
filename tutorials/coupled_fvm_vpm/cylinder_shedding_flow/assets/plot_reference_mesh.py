#!/usr/bin/env python3
"""Plot the conventional cylinder mesh at the midspan plane."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import sys

_CASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_CASE_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CASE_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


def _midspan_cells(mesh: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh["vertex_position"], dtype=float)
    connectivity = np.asarray(mesh["cell_vertex_indices"], dtype=int)
    cell_points = points[connectivity]
    centre_z = np.mean(cell_points[:, :, 2], axis=1)
    positive = centre_z[centre_z > 0.0]
    target_z = (
        float(np.min(positive)) if positive.size else float(centre_z[np.argmin(abs(centre_z))])
    )
    selected = np.isclose(centre_z, target_z, rtol=0.0, atol=1.0e-12)
    polygons = cell_points[selected, :4, :2]
    boundary_layer = np.asarray(mesh["boundary_layer_index"], dtype=int)[selected] >= 0
    return polygons, boundary_layer


def _draw_mesh(ax, polygons: np.ndarray, boundary_layer: np.ndarray) -> None:
    cartesian = PolyCollection(
        polygons[~boundary_layer],
        facecolors="none",
        edgecolors=util.COLORS["reference"],
        linewidths=0.18,
        rasterized=True,
    )
    layers = PolyCollection(
        polygons[boundary_layer],
        facecolors="none",
        edgecolors=util.COLORS["fvm"],
        linewidths=0.30,
        rasterized=True,
    )
    ax.add_collection(cartesian)
    ax.add_collection(layers)
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            0.5,
            facecolor="white",
            edgecolor=util.COLORS["accent"],
            linewidth=0.8,
            zorder=4,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")


def plot(case: Path, figure_format: str) -> Path:
    cache = case / "solution" / "reference_mesh.pkl"
    if not cache.is_file():
        raise SystemExit(f"Reference mesh cache is missing: {cache}")
    with cache.open("rb") as stream:
        payload = pickle.load(stream)
    mesh = payload["mesh"] if "mesh" in payload else payload
    polygons, boundary_layer = _midspan_cells(mesh)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=util.figure_size(15.5),
        dpi=util.FIGURE_DPI,
    )
    views = (
        ((-8.0, 20.0), (-8.0, 8.0), "Complete conventional CFD domain"),
        ((-1.0, 8.0), (-2.1, 2.1), "Cylinder and nested wake refinement"),
        ((-0.72, 0.72), (-0.72, 0.72), "Body-fitted wall-normal layers"),
    )
    for ax, (xlim, ylim, title) in zip(axes, views):
        _draw_mesh(ax, polygons, boundary_layer)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.075, top=0.96, hspace=0.48)
    output = util.save(fig, "reference_mesh", figure_format, util.FIGURE_DPI)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="png", choices=("png", "pdf"))
    parser.add_argument(
        "--case",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_flow",
    )
    args = parser.parse_args()
    plot(args.case.resolve(), args.format)


if __name__ == "__main__":
    main()
