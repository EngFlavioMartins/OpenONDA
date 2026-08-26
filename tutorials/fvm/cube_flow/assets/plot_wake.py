#!/usr/bin/env python3
"""Instantaneous wake-centreline u_x/U from the final VTU snapshot.

For the shedding regime the centreline velocity is unsteady; this plot is a
qualitative check of the near-wake recovery, not a validation quantity (the
validated quantities are St and mean Cd, see plot_forces.py).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from _common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    latest_vtu,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()

    final = latest_vtu(SOLUTION_DIR)
    if final is None:
        print(f"  WARNING: No VTU files found in {SOLUTION_DIR}")
        return
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    u = mesh.point_data.get("velocity")
    if u is None:
        print("  WARNING: No velocity field 'velocity' found in VTU.")
        return
    pts = mesh.points

    on_plane = np.abs(pts[:, 2]) < 1e-9
    if not on_plane.any():
        on_plane = pts[:, 2] < pts[:, 2].min() + 1e-9
    near_axis = np.abs(pts[:, 1]) < 0.04
    sel = on_plane & near_axis & (pts[:, 0] > 0.5)
    if not sel.any():
        print("  WARNING: No centreline points behind the cylinder.")
        return

    order = np.argsort(pts[sel, 0])
    x = pts[sel, 0][order]
    ux = u[sel, 0][order] / FREESTREAM_SPEED

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x, ux, color=COLORS["TUDdark"], linewidth=1.1)
    ax.axhline(0.0, color=COLORS["reference"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("x / D")
    ax.set_ylabel("$u_x / U_\\infty$")
    ax.set_title("Instantaneous wake centreline")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "cube_wake_centreline.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

    rev = x[ux < 0.0]
    if rev.size:
        print(f"  instantaneous reversed-flow region extends to x/D = {rev.max():.2f}")


if __name__ == "__main__":
    main()
