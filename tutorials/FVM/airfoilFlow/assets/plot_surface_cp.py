import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from _common import COLORS, build_arg_parser, figure_size, save_fig, U_INF, L_REF, RE


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
        print("  Empty mesh, skipping Cp plot")
        return
    blocks = {}
    for i in range(mesh.n_blocks):
        name = mesh[i][0] if isinstance(mesh[i], tuple) else str(i)
        blocks[name] = mesh[i]
    if "airfoil" not in blocks:
        print(f"  airfoil block not found (available: {list(blocks.keys())})")
        return
    bmesh = blocks["airfoil"]
    p = bmesh.point_data.get("p", None)
    if p is None:
        print("  No pressure data on airfoil")
        return
    rho = 1.0
    Cp = p / (0.5 * rho * U_INF**2)
    x = bmesh.points[:, 0]
    y = bmesh.points[:, 1]
    upper = y >= 0
    lower = y < 0
    x_c = x / L_REF
    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x_c[upper], Cp[upper], "o", color=COLORS["TUDdark"], markersize=3, label="Upper")
    ax.plot(x_c[lower], Cp[lower], "o", color=COLORS["AccentRed"], markersize=3, label="Lower")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"NACA0012  AoA=23$^\\circ$  Re={RE:.0f}  $C_p$ distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "airfoil_cp.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
