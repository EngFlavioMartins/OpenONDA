#!/usr/bin/env python3
"""Velocity-magnitude snapshot around the airfoil from the final VTU."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORMAPS,
    RE,
    build_arg_parser,
    figure_size,
    latest_vtu,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()

    final = latest_vtu(args.solution_dir)
    if final is None:
        print(f"  WARNING: No VTU files found in {args.solution_dir}")
        return
    print(f"  Reading: {final}")
    mesh = pv.read(final)

    u = mesh.point_data.get("velocity")
    if u is None:
        print("  No velocity data in VTU.")
        return
    pts = mesh.points
    mag = np.linalg.norm(u, axis=1)

    on_plane = np.abs(pts[:, 2]) < 1e-9
    if not on_plane.any():
        on_plane = pts[:, 2] < pts[:, 2].min() + 1e-9
    near = on_plane & (pts[:, 0] > -1.0) & (pts[:, 0] < 3.0) & (np.abs(pts[:, 1]) < 1.5)

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    sc = ax.tripcolor(
        pts[near, 0], pts[near, 1], mag[near], cmap=COLORMAPS["field_speed"], shading="gouraud"
    )
    plt.colorbar(sc, ax=ax, label="velocity magnitude / freestream speed")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(
        f"NACA 0012 velocity magnitude (Re = {RE:.0f}, $\\alpha$ = {args.angle:g}$^\\circ$)"
    )
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, "airfoil_velocity.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
