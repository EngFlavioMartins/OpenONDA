#!/usr/bin/env python3
"""
Vortex-ring tube circulation and vector-sum conservation versus t Gamma/R0^2.

Saves: figures/vortex_ring_circulation.png
"""

import matplotlib.pyplot as plt

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from ..assets.ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    centered_subplots_adjust,
    load_sampled_ring_circulation,
    load_sampled_vector_circulation_error,
    load_theme,
    plot_variants,
    save_fig,
    with_sample_gaps,
    validate_thesis_figure,
)


def line_style(st: dict, markevery: int, markersize: float | None = None) -> dict:
    """Return the shared line keywords for one stretching variant."""
    return {
        "linestyle": st["linestyle"],
        "color": st["color"],
        "lw": st["linewidth"],
        "marker": st["marker"],
        "ms": st["markersize"] if markersize is None else markersize,
        "markevery": markevery,
        "mew": st["markeredgewidth"],
    }


def main() -> None:
    args = build_arg_parser(
        "Tube-circulation estimate and vector-strength drift for all vortex-ring variants."
    ).parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig = plt.figure(figsize=(125 / 25.4, 125 / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    ax_tube = fig.add_subplot(grid[0, 0])
    ax_sum = fig.add_subplot(grid[0, 1])
    centered_subplots_adjust(fig, outer=0.15, wspace=0.55, hspace=0.64, top=0.93, bottom=0.22)
    legend_handles = []
    legend_labels = []

    zoom = fig.add_subplot(grid[1, :])
    zoom.set_xlabel(r"$t\,\Gamma_0/R_0^2$")
    zoom.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    zoom.set_xlim(0, 190)
    zoom.set_ylim(0.995, 1.0005)
    zoom.set_xticks([0, 190])
    zoom.set_yticks([0.996, 1.000])
    zoom.tick_params(pad=3)
    zoom.set_title("(a) Detail: transposed runs", pad=6)
    n_skip = 14  # plot every n-th marker

    ax_sum.set_yscale("log")
    circulation_values = []
    drift_values = []
    maximum_time = 0.0

    for variant in plot_variants():
        st = VARIANT_STYLE[variant]
        csv_path = SAMPLES_DIR / variant / "ring_diagnostics.csv"
        t, c = load_sampled_ring_circulation(csv_path)
        if t.size == 0:
            continue
        t_sum, sum_err = load_sampled_vector_circulation_error(csv_path)
        label = VARIANT_LABEL[variant]
        (line,) = ax_tube.plot(*with_sample_gaps(t, c), label=label, **line_style(st, n_skip))
        if variant in ("dns_transposed", "les_transposed"):
            zoom.plot(*with_sample_gaps(t, c), **line_style(st, n_skip, 1.5))
        circulation_values.append(c)
        maximum_time = max(maximum_time, float(t[-1]))
        if t_sum.size:
            ax_sum.semilogy(
                *with_sample_gaps(t_sum, sum_err), label=label, **line_style(st, n_skip)
            )
            drift_values.append(sum_err)
            maximum_time = max(maximum_time, float(t_sum[-1]))
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_tube, ax_sum):
        ax.set_xlabel(r"$t\,\Gamma_0/R_0^2$")
        ax.set_xlim(0.0, 1.01 * maximum_time)

    if circulation_values:
        lower = min(float(values.min()) for values in circulation_values)
        upper = max(float(values.max()) for values in circulation_values)
        padding = max(0.001, 0.15 * (upper - lower))
        ax_tube.set_ylim(lower - padding, upper + padding)
    if drift_values:
        lower = min(float(values.min()) for values in drift_values)
        upper = max(float(values.max()) for values in drift_values)
        ax_sum.set_ylim(0.7 * lower, 1.4 * upper)

    ax_tube.set_title(r"(a) Tube circulation")
    ax_tube.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    ax_sum.set_title(r"(b) Vector drift")
    ax_sum.set_ylabel(r"$e_{\Gamma}$")

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    validate_thesis_figure(fig, (ax_tube, ax_sum, zoom))
    save_fig(
        fig,
        figs / "vortex_ring_circulation.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
