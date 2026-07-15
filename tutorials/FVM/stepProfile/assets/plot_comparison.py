#!/usr/bin/env python3
"""Plot the stepped geometry, velocity field, and reattachment history."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORMAPS,
    COLORS,
    REFERENCES,
    build_arg_parser,
    load_csv_columns,
    save_fig,
)


def main():
    args = build_arg_parser().parse_args()
    fields = load_csv_columns(Path(args.solution_dir) / "fields.csv")
    history = load_csv_columns(Path(args.solution_dir) / "reattachment_history.csv")
    if not fields or not history:
        return
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    x, y = fields["x_over_h"], fields["y_over_h"]
    speed = np.hypot(fields["u"], fields["v"])
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
    x_re = history["x_reattachment_over_h"]
    finite = np.isfinite(x_re)
    history_ax.plot(time[finite], x_re[finite], linewidth=1.2)
    ref = REFERENCES.get(args.Re, {})
    if "x_r" in ref:
        history_ax.axhspan(
            *ref["x_r"],
            color=COLORS["reference"],
            alpha=0.25,
            label=f"Armaly et al. 1983: {ref['x_r'][0]:.1f}-{ref['x_r'][1]:.1f}",
        )
        history_ax.legend(loc="lower right", fontsize=8)
    history_ax.set_xlabel(r"$t U_b/h$")
    history_ax.set_ylabel(r"$x_r/h$")
    history_ax.set_title("Resolved near-wall reattachment estimate")
    history_ax.grid(True, alpha=0.3)
    if finite.any():
        x_final = float(x_re[finite][-1])
        print(f"  final x_r/h = {x_final:.2f}", end="")
        if "x_r" in ref:
            lo, hi = ref["x_r"]
            print(
                f"  [Armaly et al. {lo:.1f}-{hi:.1f}: "
                f"{'OK' if lo <= x_final <= hi else 'OUT OF BAND'}]",
                end="",
            )
        print()
    save_fig(fig, "step_comparison.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)


if __name__ == "__main__":
    main()
