#!/usr/bin/env python3
"""Velocity-magnitude snapshot around the airfoil from the final VTU."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pyvista as pv

from ._common import (  # noqa: E402
    COLORMAPS,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    RE,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    latest_vtu,
    save_fig,
)


def _section_polygons_and_speed(mesh):
    """Cut native FVM cells at the midspan without averaging cell values."""
    if "velocity" not in mesh.cell_data:
        if "velocity" not in mesh.point_data:
            raise ValueError("Snapshot has no velocity field")
        mesh = mesh.point_data_to_cell_data()
    section = mesh.slice(normal="z", origin=(0.0, 0.0, 0.0))
    velocity = section.cell_data.get("velocity")
    if section.n_cells == 0 or velocity is None:
        raise ValueError("Snapshot has no velocity-bearing cells at midspan")

    faces = section.faces
    polygons = []
    cursor = 0
    while cursor < len(faces):
        n = int(faces[cursor])
        if n < 3:
            raise ValueError("Midspan section contains a non-polygon cell")
        polygons.append(section.points[faces[cursor + 1 : cursor + 1 + n], :2])
        cursor += n + 1
    if len(polygons) != section.n_cells:
        raise ValueError("Midspan section polygon and field counts differ")
    return polygons, np.linalg.norm(velocity, axis=1) / FREESTREAM_SPEED


def main():
    args = build_arg_parser().parse_args()

    final = latest_vtu(SOLUTION_DIR)
    if final is None:
        raise SystemExit(f"  WARNING: No VTU files found in {SOLUTION_DIR}")
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    polygons, speed = _section_polygons_and_speed(mesh)

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    sc = PolyCollection(polygons, cmap=COLORMAPS["field_speed"], edgecolors="none")
    sc.set_array(speed)
    ax.add_collection(sc)
    plt.colorbar(sc, ax=ax, label="velocity magnitude / freestream speed")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(
        f"NACA 0012 velocity magnitude (Re = {RE:.0f}, $\\alpha$ = {args.angle:g}$^\\circ$)"
    )
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, "airfoil_velocity.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
