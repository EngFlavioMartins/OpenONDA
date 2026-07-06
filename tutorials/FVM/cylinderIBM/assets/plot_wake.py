#!/usr/bin/env python3
"""Wake centreline u_x(x) from the final VTU snapshot + recirculation length.

The recirculation length L (from the rear stagnation point of the cylinder to
the point where centreline u_x changes sign back to positive) is the second
quality monitor for the steady Re = 30 case: reference L/D = 1.55-1.70
(Constant et al. 2017, Table 2)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS, D_REF, REFERENCES, U_INF, build_arg_parser, figure_size, save_fig,
)


def recirculation_length(x, u, D=D_REF):
    """Distance from the cylinder rear (x = D/2) to the u_x sign recovery."""
    rear = 0.5 * D
    mask = x > rear
    xs, us = x[mask], u[mask]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = us < 0.0
    if not neg.any():
        return None
    # Last negative sample, then linear interpolation to the zero crossing.
    i_last = np.where(neg)[0][-1]
    if i_last + 1 >= len(xs):
        return None
    x0, x1 = xs[i_last], xs[i_last + 1]
    u0, u1 = us[i_last], us[i_last + 1]
    x_zero = x0 - u0 * (x1 - x0) / (u1 - u0)
    return float(x_zero - rear)


def main():
    args = build_arg_parser().parse_args()
    ref = REFERENCES.get(args.Re, {})

    vtu_files = sorted(Path(args.solution_dir).glob("*.vtu"))
    if not vtu_files:
        print(f"  WARNING: no VTU files in {args.solution_dir}")
        return
    final = vtu_files[-1]
    print(f"  Reading: {final.name}")
    mesh = pv.read(str(final))
    cc = mesh.cell_centers().points
    u = mesh.cell_data.get("U")
    if u is None:
        mesh = mesh.point_data_to_cell_data()
        u = mesh.cell_data.get("U")
    if u is None:
        print("  WARNING: no velocity field 'U' in VTU.")
        return

    # Centreline: cells nearest to y = 0 (one row on this rectilinear mesh).
    y_vals = np.unique(np.round(cc[:, 1], 10))
    y_row = y_vals[np.argmin(np.abs(y_vals))]
    mask = np.isclose(cc[:, 1], y_row)
    x_cl = cc[mask, 0]
    u_cl = u[mask, 0]
    order = np.argsort(x_cl)
    x_cl, u_cl = x_cl[order], u_cl[order]

    L = recirculation_length(x_cl, u_cl)

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x_cl / D_REF, u_cl / U_INF, color=COLORS["TUDdark"], linewidth=1.0)
    ax.axhline(0.0, color=COLORS["AxisBlack"], linewidth=0.5)
    ax.axvspan(-0.5, 0.5, color=COLORS["MaskGray"], label="cylinder")
    if L is not None and "L_over_D" in ref:
        lo, hi = ref["L_over_D"]
        ax.axvspan(0.5 + lo, 0.5 + hi, color=COLORS["reference"], alpha=0.3,
                   label=f"ref. wake closure: L/D = {lo:.2f}-{hi:.2f}")
    if L is not None:
        ax.axvline(0.5 + L, color=COLORS["TUDred"],
                   linestyle="--", linewidth=0.8, label=f"L/D = {L / D_REF:.3f}")
    ax.set_xlim(-2, 10)
    ax.set_xlabel("x / D")
    ax.set_ylabel(r"$u_x / U_\infty$")
    ax.set_title(f"Wake centreline (Re = {args.Re:g})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "wake_centreline.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)

    if L is not None:
        msg = f"  recirculation length L/D = {L / D_REF:.3f}"
        if "L_over_D" in ref:
            lo, hi = ref["L_over_D"]
            status = "OK" if lo * 0.9 <= L / D_REF <= hi * 1.1 else "OUT OF BAND"
            msg += f"  [reference {lo:.2f}-{hi:.2f}: {status}]"
        print(msg)
    else:
        print("  no recirculation detected (unsteady case or too early).")


if __name__ == "__main__":
    main()
