#!/usr/bin/env python3
"""Plot the stepped geometry, velocity field, and reattachment history."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from _common import (  # noqa: E402
    COLORMAPS,
    FIGURES_DIR,
    SOLUTION_DIR,
    build_arg_parser,
    load_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    fields = load_csv_columns(Path(SOLUTION_DIR) / "fields.csv")
    history = load_csv_columns(Path(SOLUTION_DIR) / "reattachment_history.csv")
    if not fields or not history:
        return
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    x, y = fields["position_x_over_height"], fields["position_y_over_height"]
    speed = np.hypot(fields["velocity_x"], fields["velocity_y"])
    triangulation = mtri.Triangulation(x, y)
    triangle_centres = np.column_stack(
        (x[triangulation.triangles].mean(axis=1), y[triangulation.triangles].mean(axis=1))
    )
    triangulation.set_mask((triangle_centres[:, 0] < 0.0) & (triangle_centres[:, 1] < 1.0))

    fig, (flow_ax, history_ax) = plt.subplots(2, 1, figsize=(7.1, 5.8), constrained_layout=True)
    contour = flow_ax.tricontourf(triangulation, speed, levels=30, cmap=COLORMAPS["field_speed"])
    flow_ax.plot([-4.0, 0.0, 0.0, 20.0], [1.0, 1.0, 0.0, 0.0], color="black", linewidth=1.5)
    flow_ax.plot([-4.0, 20.0], [2.0, 2.0], color="black", linewidth=1.5)
    flow_ax.set_xlim(-4.0, 20.0)
    flow_ax.set_ylim(-0.05, 2.05)
    flow_ax.set_aspect("equal", adjustable="box")
    flow_ax.set_xlabel(r"$x/h$")
    flow_ax.set_ylabel(r"$y/h$")
    flow_ax.set_title("Backward-facing-step speed and geometric outline")
    fig.colorbar(contour, ax=flow_ax, label=r"$|U|/U_b$")

    time = history["time"]
    x_re = history["reattachment_position_over_height"]
    finite = np.isfinite(x_re)
    history_ax.plot(time[finite], x_re[finite], linewidth=1.2)
    history_ax.set_xlabel(r"$t U_b/h$")
    history_ax.set_ylabel(r"$x_r/h$")
    history_ax.set_title("Resolved near-wall reattachment estimate")
    history_ax.grid(True, alpha=0.3)
    if finite.any():
        x_final = float(x_re[finite][-1])
        print(f"  final x_r/h = {x_final:.2f}")
    save_fig(fig, "step_comparison.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
