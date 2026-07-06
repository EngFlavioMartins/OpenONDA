import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from _common import COLORMAPS, build_arg_parser, figure_size, save_fig, RE


def main():
    args = build_arg_parser().parse_args()
    solution_dir = args.solution_dir
    pvd_path = os.path.join(solution_dir, "airfoil.pvd")
    if not os.path.exists(pvd_path):
        print(f"  PVD not found: {pvd_path}")
        return
    try:
        mesh = pv.read(pvd_path)
    except Exception as e:
        print(f"  Error reading PVD: {e}")
        return
    if not mesh.n_cells:
        print("  Empty mesh, skipping velocity plot")
        return
    center = mesh.cell_centers()
    U = mesh.point_data.get("U", None)
    if U is None:
        try:
            cell_U = mesh.cell_data.get("U", None)
            if cell_U is not None:
                U_vec = cell_U
                points = center.points
            else:
                print("  No velocity data")
                return
        except Exception:
            print("  No velocity data")
            return
    else:
        U_vec = U
        points = mesh.points
    mag = np.linalg.norm(U_vec, axis=1)
    x = points[:, 0]
    y = points[:, 1]
    near = (x >= -1.0) & (x <= 3.0) & (y >= -1.5) & (y <= 1.5)
    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    sc = ax.scatter(x[near], y[near], c=mag[near], s=2, cmap=COLORMAPS["field_speed"], alpha=0.7)
    plt.colorbar(sc, ax=ax, label="$|U|$ [m/s]")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(f"NACA0012  AoA=23$^\\circ$  Re={RE:.0f}  Velocity magnitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, "airfoil_velocity.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
