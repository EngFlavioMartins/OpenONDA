#!/usr/bin/env python3
"""Vorticity of the locked-in wake behind the heaving cylinder (final frame)."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from openonda.plotting import centered_subplots_adjust

from assets._common import THEME, build_arg_parser, figure_size, latest_fvm_frame, save_fig
from setup import AMPLITUDE, DIAMETER, FREQUENCY


def main():
    args = build_arg_parser().parse_args()
    time, frame = latest_fvm_frame()
    grid = pv.read(frame)
    centres = grid.cell_centers().points
    vorticity = grid.cell_data["vorticity"][:, 2]

    # The fields are in the inertial frame; the cylinder has moved to y_c(t).
    centre_y = AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY * time)
    x, y = centres[:, 0] / DIAMETER, centres[:, 1] / DIAMETER
    shown = (x > -2.0) & (x < 16.0) & (np.abs(y) < 4.0)
    # Colour range of the wake; the thin wall layers saturate
    limit = np.percentile(np.abs(vorticity[shown & (x > 1.0)]), 99)

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    image = ax.tripcolor(
        x[shown],
        y[shown],
        vorticity[shown],
        cmap=THEME.COLORMAPS["vorticity"],
        vmin=-limit,
        vmax=limit,
        shading="gouraud",
    )
    ax.add_patch(
        mpatches.Circle((0.0, centre_y / DIAMETER), 0.5, facecolor="0.2", edgecolor="k", zorder=5)
    )
    fig.colorbar(image, ax=ax, location="top", shrink=0.6, label=r"$\omega_z$ [1/s]")
    ax.text(0.98, 0.04, f"t = {time:.1f} s", transform=ax.transAxes, ha="right")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_aspect("equal")
    centered_subplots_adjust(fig, outer=0.12, bottom=0.18, top=0.80)
    save_fig(fig, "oscillating_cylinder_vorticity.png", args.format, args.dpi)


if __name__ == "__main__":
    main()
