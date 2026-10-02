#!/usr/bin/env python3
"""Signed z-vorticity snapshot of the von Karman street from the final VTU."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np
import pyvista as pv

from ._common import (
    THEME,  # noqa: E402
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    latest_vtu,
    save_fig,
    snapshot_vector_field,
)


def main():
    args = build_arg_parser().parse_args()

    final = latest_vtu(SOLUTION_DIR)
    if final is None:
        raise SystemExit(f"  WARNING: No VTU files found in {SOLUTION_DIR}")
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    vort, pts = snapshot_vector_field(mesh, "vorticity")
    wz = vort[:, 2]

    # Quasi-2D mesh: keep one physical z plane of field locations.
    mask = np.abs(pts[:, 2]) < 1e-9
    if not mask.any():
        mask = pts[:, 2] < pts[:, 2].min() + 1e-9
    x, y, wz = pts[mask, 0], pts[mask, 1], wz[mask]

    # Crop to the street; symmetric levels show the alternating sign.
    crop = (x > -3.0) & (x < 15.0) & (np.abs(y) < 4.0)
    x, y, wz = x[crop], y[crop], wz[crop]
    lim = np.percentile(np.abs(wz), 98)

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    sc = ax.tripcolor(
        x, y, wz, cmap=THEME.COLORMAPS["vorticity"], vmin=-lim, vmax=lim, shading="gouraud"
    )
    ax.add_patch(
        mpatches.Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="0.2", edgecolor="k", zorder=5)
    )
    plt.colorbar(sc, ax=ax, label=r"$\omega_z$ [1/s]")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_aspect("equal")

    centered_subplots_adjust(plt.gcf(), outer=.16, bottom=.22, top=.96)
    save_fig(fig, "cube_vorticity_street.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
