#!/usr/bin/env python3
"""Plot |ω| slice at z=0 from the final VTU snapshot."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import COLORMAPS, build_arg_parser, figure_size, save_fig


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

    u = mesh.point_data.get("U", None)
    if u is None:
        print("  WARNING: No velocity field 'U' found in VTU.")
        return

    vort = mesh.point_data.get("vorticity", None)
    if vort is None:
        print("  WARNING: No 'vorticity' field found in VTU.")
        return

    vort_mag = np.linalg.norm(vort, axis=1)

    z_tol = 0.05
    mask = np.abs(mesh.points[:, 2]) < z_tol
    if not mask.any():
        print("  WARNING: No points near z=0 found.")
        return

    slice_pts = mesh.points[mask]
    slice_vort = vort_mag[mask]

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    sc = ax.scatter(
        slice_pts[:, 0],
        slice_pts[:, 1],
        c=slice_vort,
        s=1.0,
        cmap=COLORMAPS["vorticity_magnitude"],
        vmin=0,
        vmax=np.percentile(slice_vort, 95),
    )
    plt.colorbar(sc, ax=ax, label=r"$|\omega|$ [1/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(r"$|\omega|$ at $z=0$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "cube_vorticity_slice.png", figures_dir, dpi=dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
