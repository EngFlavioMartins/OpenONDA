#!/usr/bin/env python3
"""Vorticity and velocity-magnitude snapshots with the IBM marker overlay."""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import numpy as np
import pyvista as pv

from openonda.results import hexahedron_footprints, latest_fvm_frame, snapshot_cell_field

from ._common import (  # noqa: E402
    COLORS,
    COLORMAPS,
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    load_markers,
    save_fig,
)


def _cell_footprints(mesh):
    return hexahedron_footprints(mesh)


def main():
    args = build_arg_parser().parse_args()

    final = latest_fvm_frame(SOLUTION_DIR)
    print(f"  Reading: {final.name}")
    mesh = pv.read(str(final))
    footprints = _cell_footprints(mesh)
    u = snapshot_cell_field(mesh, "velocity")
    vort = snapshot_cell_field(mesh, "vorticity") if "vorticity" in mesh.array_names else None

    markers = load_markers(SOLUTION_DIR)

    fields = [
        (
            "field_velocity_magnitude.png",
            r"$|\mathbf{u}|/U_\infty$",
            np.linalg.norm(u, axis=1),
            COLORMAPS["field_speed"],
            None,
        )
    ]
    if vort is not None:
        wz = vort[:, 2] if vort.ndim == 2 else vort
        lim = max(np.percentile(np.abs(wz), 99.0), 1e-12)
        fields.append(
            (
                "field_vorticity.png",
                r"$\omega_z$ [s$^{-1}$]",
                wz,
                COLORMAPS["vorticity"],
                (-lim, lim),
            )
        )

    for name, label, values, cmap, clim in fields:
        fig = plt.figure(figsize=(125 / 25.4, 78 / 25.4))
        ax = fig.add_axes([0.110, 0.305, 0.780, 0.673077])
        image = PolyCollection(footprints, cmap=cmap, edgecolors="none", antialiased=False)
        image.set_array(values)
        if clim is not None:
            image.set_clim(*clim)
        ax.add_collection(image)
        ax.add_patch(
            Polygon(
                markers[:, :2],
                closed=True,
                facecolor="white",
                edgecolor=COLORS["black"],
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.set_xlim(-3, 10)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
        ax.tick_params(axis="y", pad=8)
        cax = fig.add_axes([0.110, 0.15, 0.780, 0.025])
        fig.colorbar(image, cax=cax, orientation="horizontal", label=label)
        save_fig(fig, name, FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
