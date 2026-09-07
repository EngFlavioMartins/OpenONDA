#!/usr/bin/env python3
"""Plot the time reached by each vortex-stretching formulation."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tutorials.vpm.vortex_ring.assets.ring_metrics import (
    FIGURES_DIR,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_stability_results,
    load_theme,
    save_fig,
    with_sample_gaps,
)


def main() -> None:
    args = build_arg_parser("Vortex-ring instability time.").parse_args()
    load_theme()
    results = load_stability_results()
    if not results:
        raise SystemExit("No vortex-ring results are available")

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    fig.subplots_adjust(left=0.32, right=0.96, top=0.84, bottom=0.37)

    times = [result["normalized_time"] for result in results]
    positions = list(range(len(results)))
    colors = [VARIANT_STYLE[result["variant"]]["color"] for result in results]
    ax.barh(positions, times, color=colors, alpha=0.42, height=0.56)

    markers = {
        "instability_detected": "X",
        "resolution_lost": "X",
        "horizon_reached": ">",
        "complete": ">",
    }
    for position, result in enumerate(results):
        time = result["normalized_time"]
        color = VARIANT_STYLE[result["variant"]]["color"]
        ax.plot(
            time,
            position,
            marker=markers.get(result["status"], "o"),
            color=color,
            markersize=7,
            linestyle="none",
        )
        ax.annotate(
            f"{float(format(time, '.2g')):g}",
            (time, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.95,
        )

    ax.set_yticks(positions, [VARIANT_LABEL[result["variant"]] for result in results])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(times) * 1.18 if max(times) > 0.0 else 1.0)
    ax.set_xlabel(r"$t\,\Gamma_0/R_0^2$")
    ax.set_title("Time reached")
    ax.grid(axis="y", visible=False)
    fig.legend(
        handles=(
            Line2D([], [], marker="X", linestyle="none", color="0.25", label="Limit reached"),
            Line2D([], [], marker=">", linestyle="none", color="0.25", label="End time"),
            Line2D([], [], marker="o", linestyle="none", color="0.25", label="Running"),
        ),
        ncol=2,
        fontsize=10.95,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_fig(
        fig,
        FIGURES_DIR / "vortex_ring_stability.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
