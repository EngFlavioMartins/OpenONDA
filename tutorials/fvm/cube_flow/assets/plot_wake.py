#!/usr/bin/env python3
"""Instantaneous wake-centreline u_x/U from the final VTU snapshot.

For the shedding regime the centreline velocity is unsteady; this plot is a
qualitative check of the near-wake recovery, not a validation quantity (the
validated quantities are St and mean Cd, see plot_forces.py).
"""

import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np
import pyvista as pv

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    latest_fvm_frame,
    save_fig,
    snapshot_vector_field,
)


def main():
    args = build_arg_parser().parse_args()

    final = latest_fvm_frame(SOLUTION_DIR)
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    u, pts = snapshot_vector_field(mesh, "velocity")

    z_planes = np.unique(pts[:, 2])
    z_plane = z_planes[np.argmin(np.abs(z_planes))]
    on_plane = np.isclose(pts[:, 2], z_plane)
    near_axis = np.abs(pts[:, 1]) < 0.04
    sel = on_plane & near_axis & (pts[:, 0] > 0.5)

    order = np.argsort(pts[sel, 0])
    x = pts[sel, 0][order]
    ux = u[sel, 0][order] / FREESTREAM_SPEED

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x, ux, color=COLORS["fvm"], linewidth=1.1)
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("x / D")
    ax.set_ylabel("$u_x / U_\\infty$")
    ax.grid(False)

    centered_subplots_adjust(plt.gcf(), outer=0.16, bottom=0.20, top=0.96)
    save_fig(fig, "cube_wake_centreline.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

    rev = x[ux < 0.0]
    if rev.size:
        print(f"  instantaneous reversed-flow region extends to x/D = {rev.max():.2f}")


if __name__ == "__main__":
    main()
