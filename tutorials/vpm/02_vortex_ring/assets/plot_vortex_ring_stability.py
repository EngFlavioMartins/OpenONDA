#!/usr/bin/env python3
"""Plot the time reached by each vortex-stretching formulation."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from ..assets.ring_metrics import (
    FIGURES_DIR,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    centered_subplots_adjust,
    figure_size,
    load_stability_results,
    load_theme,
    save_fig,
    with_sample_gaps,
    validate_thesis_figure,
)


def main() -> None:
    args = build_arg_parser("Vortex-ring instability time.").parse_args()
    load_theme()
    results = load_stability_results()
    if not results:
        raise SystemExit("No vortex-ring results are available")

    fig, ax = plt.subplots(figsize=figure_size("wide_short"))
    centered_subplots_adjust(fig, outer=0.10, top=0.84, bottom=0.34)

    times = [result["normalized_time"] for result in results]
    positions = list(range(len(results)))
    colors = [VARIANT_STYLE[result["variant"]]["color"] for result in results]
    ax.barh(positions, times, color=colors, alpha=0.42, height=0.56)

    maximum_time = max(times)
    label_space = 0.40 * maximum_time if maximum_time > 0.0 else 0.40
    right_space = 0.18 * maximum_time if maximum_time > 0.0 else 0.18
    x_min = -label_space
    x_max = maximum_time + right_space

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
        )

    ax.set_yticks(positions, [""] * len(positions))
    for position, result in enumerate(results):
        ax.text(
            x_min + 0.025 * (x_max - x_min),
            position,
            VARIANT_LABEL[result["variant"]],
            ha="left",
            va="center",
        )
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    nonnegative_ticks = [tick for tick in ax.get_xticks() if tick >= 0.0]
    ax.set_xticks(nonnegative_ticks)
    ax.axvline(0.0, color="0.35", linewidth=0.5)
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
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    validate_thesis_figure(fig, ax)
    save_fig(
        fig,
        FIGURES_DIR / "vortex_ring_stability.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
