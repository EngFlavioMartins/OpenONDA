#!/usr/bin/env python3
"""Velocity-magnitude snapshot around the airfoil from the final VTU."""

import matplotlib.pyplot as plt
from openonda.results import section_polygons
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
    latest_fvm_frame,
    save_fig,
)


def _section_polygons_and_speed(mesh):
    polygons, velocity = section_polygons(
        mesh, normal="z", origin=(0.0, 0.0, 0.0), field="velocity"
    )
    return polygons, np.linalg.norm(velocity, axis=1) / FREESTREAM_SPEED


def main():
    args = build_arg_parser().parse_args()

    final = latest_fvm_frame(SOLUTION_DIR)
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    polygons, speed = _section_polygons_and_speed(mesh)

    fig = plt.figure(figsize=(125 / 25.4, 105 / 25.4))
    ax = fig.add_axes([0.132, 0.32, 0.736, 0.657143])
    sc = PolyCollection(polygons, cmap=COLORMAPS["field_speed"], edgecolors="none")
    sc.set_array(speed)
    ax.add_collection(sc)
    cax = fig.add_axes([0.132, 0.14, 0.736, 0.022])
    fig.colorbar(
        sc, cax=cax, orientation="horizontal", label="velocity magnitude / freestream speed"
    )
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_aspect("equal")
    ax.tick_params(axis="y", pad=8)

    save_fig(fig, "airfoil_velocity.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
