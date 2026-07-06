#!/usr/bin/env python3
"""Plot wake centreline u_x from the final VTU snapshot."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, build_arg_parser, figure_size, save_fig, U_INF


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    solution_dir = args.solution_dir
    figures_dir = args.figures_dir
    dpi = args.dpi

    vtu_dir = Path(solution_dir)
    vtu_files = sorted(vtu_dir.glob("*.vtu"))
    if not vtu_files:
        # Try VTK subdirectory (legacy)
        vtu_dir = Path(solution_dir) / "VTK"
        if vtu_dir.exists():
            vtu_files = sorted(vtu_dir.glob("*.vtu"))
    if not vtu_files:
        print(f"  WARNING: No VTU files found in {solution_dir}")
        return
    if not vtu_files:
        print(f"  WARNING: No VTU files found in {vtu_dir}")
        return

    final_vtu = vtu_files[-1]
    print(f"  Reading: {final_vtu}")

    mesh = pv.read(str(final_vtu))

    points = mesh.points
    u = mesh.point_data.get("U", None)
    if u is None:
        print("  WARNING: No velocity field 'U' found in VTU.")
        return

    y_tol = 0.05
    z_tol = 0.05

    mask = (
        (np.abs(points[:, 1]) < y_tol)
        & (np.abs(points[:, 2]) < z_tol)
        & (points[:, 0] >= 0.0)
    )
    if not mask.any():
        print("  WARNING: No centreline points found.")
        return

    x_cl = points[mask, 0]
    ux_cl = u[mask, 0]
    sort_idx = np.argsort(x_cl)
    x_cl = x_cl[sort_idx]
    ux_cl = ux_cl[sort_idx]

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(
        x_cl,
        ux_cl / U_INF,
        color=COLORS["TUDdark"],
        linestyle="-",
        linewidth=0.8,
        label=f"t = {mesh.field_data.get('Time', '?')}",
    )

    face_min = ux_cl.min()
    ax.axhline(0, color=COLORS["reference"], linestyle=":", linewidth=0.5)
    ax.text(
        0.97, 0.92, f"+x face min u_x/U∞ = {face_min / U_INF:.3f}",
        transform=ax.transAxes, ha="right", va="top" ,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=COLORS["LightText"],
            edgecolor=COLORS["background_light"],
            alpha=0.8,
        ),
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("$u_x / U_\\infty$")
    ax.set_title("Wake centreline velocity (y=0, z=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "cube_wake_centerline.png", figures_dir, dpi=dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
