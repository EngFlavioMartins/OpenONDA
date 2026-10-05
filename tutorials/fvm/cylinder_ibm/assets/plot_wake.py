#!/usr/bin/env python3
"""Wake centreline u_x(x) from the final VTU snapshot + recirculation length.

The recirculation length L (from the rear stagnation point of the cylinder to
the point where centreline u_x changes sign back to positive) is the second
quality monitor for the steady Re = 30 case: reference L/D = 1.55-1.70
(Constant et al. 2017, Table 2)."""

import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np
import pyvista as pv

from openonda.results import latest_fvm_frame, snapshot_cell_field

from ._common import (  # noqa: E402
    COLORS,
    D_REF,
    FIGURES_DIR,
    REFERENCES,
    FREESTREAM_SPEED,
    SOLUTION_DIR,
    build_arg_parser,
    figure_size,
    save_fig,
)


def recirculation_length(x, u, D=D_REF, center_x=0.0):
    """Distance from the cylinder rear to the u_x sign recovery."""
    rear = center_x + 0.5 * D
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


def _wake_endpoint_over_d(length, D, center_x=0.0):
    return (center_x + 0.5 * D + length) / D


def main():
    args = build_arg_parser().parse_args()
    ref = REFERENCES.get(args.Re, {})

    final = latest_fvm_frame(SOLUTION_DIR)
    print(f"  Reading: {final.name}")
    mesh = pv.read(str(final))
    cell_centre = mesh.cell_centers().points
    u = snapshot_cell_field(mesh, "velocity")

    # Centreline: cells nearest to y = 0 (one row on this rectilinear mesh).
    y_vals = np.unique(np.round(cell_centre[:, 1], 10))
    y_row = y_vals[np.argmin(np.abs(y_vals))]
    mask = np.isclose(cell_centre[:, 1], y_row)
    x_cl = cell_centre[mask, 0]
    u_cl = u[mask, 0]
    order = np.argsort(x_cl)
    x_cl, u_cl = x_cl[order], u_cl[order]

    L = recirculation_length(x_cl, u_cl)

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(x_cl / D_REF, u_cl / FREESTREAM_SPEED, color=COLORS["fvm"], linewidth=1.0)
    ax.axhline(0.0, color=COLORS["black"], linewidth=0.6)
    ax.axvspan(-0.5, 0.5, color=COLORS["light_gray"], label="cylinder")
    if L is not None and "L_over_D" in ref:
        lo, hi = ref["L_over_D"]
        ax.axvspan(
            0.5 + lo,
            0.5 + hi,
            color=COLORS["reference"],
            alpha=0.3,
            label=f"ref. wake closure: L/D = {lo:.2g}-{hi:.2g}",
        )
    if L is not None:
        ax.axvline(
            _wake_endpoint_over_d(L, D_REF),
            color=COLORS["fvm"],
            linestyle="-",
            linewidth=0.8,
            label=f"L/D = {L / D_REF:.2g}",
        )
    ax.set_xlim(-2, 10)
    ax.set_xlabel("x / D")
    ax.set_ylabel(r"$u_x / U_\infty$")
    ax.legend()
    ax.grid(False)
    centered_subplots_adjust(fig, outer=0.100, bottom=0.20, top=0.984)
    save_fig(fig, "wake_centreline.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)

    if L is not None:
        msg = f"  recirculation length L/D = {L / D_REF:.3f}"
        if "L_over_D" in ref:
            lo, hi = ref["L_over_D"]
            status = "OK" if lo <= L / D_REF <= hi else "OUT OF BAND"
            msg += f"  [reference {lo:.2f}-{hi:.2f}: {status}]"
        print(msg)
    else:
        print("  no recirculation detected (unsteady case or too early).")


if __name__ == "__main__":
    main()
