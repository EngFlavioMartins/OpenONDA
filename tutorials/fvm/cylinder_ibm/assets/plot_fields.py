#!/usr/bin/env python3
"""Vorticity and velocity-magnitude snapshots with the IBM marker overlay."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORS, COLORMAPS, build_arg_parser, figure_size, load_markers, save_fig  # noqa: E402


def main():
    args = build_arg_parser().parse_args()

    vtu_files = sorted(Path(args.solution_dir).glob("*.vtu"))
    if not vtu_files:
        print(f"  WARNING: no VTU files in {args.solution_dir}")
        return
    final = vtu_files[-1]
    print(f"  Reading: {final.name}")
    mesh = pv.read(str(final))
    cc = mesh.cell_centres().points
    u = mesh.cell_data.get("velocity")
    vort = mesh.cell_data.get("vorticity")
    if u is None:
        mesh = mesh.point_data_to_cell_data()
        u = mesh.cell_data.get("velocity")
        vort = mesh.cell_data.get("vorticity")

    markers = load_markers(args.solution_dir)

    fields = [("velocity_magnitude", np.linalg.norm(u, axis=1), COLORMAPS["field_speed"], None)]
    if vort is not None:
        wz = vort[:, 2] if vort.ndim == 2 else vort
        lim = max(np.percentile(np.abs(wz), 99.0), 1e-12)
        fields.append((r"$\omega_z$", wz, COLORMAPS["vorticity"], (-lim, lim)))

    for label, f, cmap, clim in fields:
        fig, ax = plt.subplots(figsize=figure_size("wide"))
        sc = ax.scatter(cc[:, 0], cc[:, 1], c=f, s=2.5, cmap=cmap, marker="s", linewidths=0)
        if clim is not None:
            sc.set_clim(*clim)
        if markers is not None:
            ax.plot(
                np.append(markers[:, 0], markers[0, 0]),
                np.append(markers[:, 1], markers[0, 1]),
                "-",
                color=COLORS["AxisBlack"],
                linewidth=0.8,
            )
        ax.set_xlim(-3, 10)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.tight_layout()
        name = (
            "field_velocity_magnitude.png"
            if label == "velocity_magnitude"
            else "field_vorticity.png"
        )
        save_fig(fig, name, args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
