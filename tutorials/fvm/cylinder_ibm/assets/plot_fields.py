#!/usr/bin/env python3
"""Vorticity and velocity-magnitude snapshots with the IBM marker overlay."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import numpy as np
import pyvista as pv

from openonda.plotting import latest_fvm_snapshot

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
    """Return the lower face of each hexahedral cell in the x-y plane."""
    if mesh.n_cells == 0 or not np.all(mesh.celltypes == pv.CellType.HEXAHEDRON):
        raise ValueError("Field snapshot must contain hexahedral cells")
    cell_points = mesh.points[mesh.cells.reshape(-1, 9)[:, 1:]]
    lower = np.argsort(cell_points[:, :, 2], axis=1)[:, :4]
    face = cell_points[np.arange(mesh.n_cells)[:, None], lower, :2]
    center = face.mean(axis=1, keepdims=True)
    angle = np.arctan2(face[:, :, 1] - center[:, :, 1], face[:, :, 0] - center[:, :, 0])
    return face[np.arange(mesh.n_cells)[:, None], np.argsort(angle, axis=1)]


def main():
    args = build_arg_parser().parse_args()

    final = latest_fvm_snapshot(SOLUTION_DIR)
    if final is None:
        raise SystemExit(f"  No field snapshots in {SOLUTION_DIR}")
    print(f"  Reading: {final.name}")
    mesh = pv.read(str(final))
    footprints = _cell_footprints(mesh)
    u = mesh.cell_data.get("velocity")
    vort = mesh.cell_data.get("vorticity")
    if u is None:
        mesh = mesh.point_data_to_cell_data()
        u = mesh.cell_data.get("velocity")
        vort = mesh.cell_data.get("vorticity")

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
        ax = fig.add_axes([.110, .305, .780, .673077])
        image = PolyCollection(footprints, cmap=cmap, edgecolors="none", antialiased=False)
        image.set_array(values)
        if clim is not None:
            image.set_clim(*clim)
        ax.add_collection(image)
        if markers is not None:
            ax.add_patch(
                Polygon(
                    markers[:, :2],
                    closed=True,
                    facecolor="white",
                    edgecolor=COLORS["AxisBlack"],
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
        cax = fig.add_axes([.110, .15, .780, .025])
        fig.colorbar(image, cax=cax, orientation="horizontal", label=label)
        save_fig(fig, name, FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
