#!/usr/bin/env python3
"""
Vortex-ring tube circulation and vector-sum conservation versus t Gamma/R0^2.

Saves: figures/vortex_ring_circulation.png
"""

import matplotlib.pyplot as plt

from tutorials.vpm.vortex_ring.assets.ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_sampled_ring_circulation,
    load_sampled_vector_circulation_error,
    load_theme,
    plot_variants,
    save_fig,
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

    fig, (ax_tube, ax_sum) = plt.subplots(1, 2, figsize=figure_size("wide_short"), sharex=True)
    fig.subplots_adjust(wspace=0.48, left=0.12, right=0.97, top=0.88, bottom=0.28)
    legend_handles = []
    legend_labels = []

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
        (line,) = ax_tube.plot(t, c, label=label, **line_style(st, n_skip))
        circulation_values.append(c)
        maximum_time = max(maximum_time, float(t[-1]))
        if t_sum.size:
            ax_sum.semilogy(t_sum, sum_err, label=label, **line_style(st, n_skip))
            drift_values.append(sum_err)
            maximum_time = max(maximum_time, float(t_sum[-1]))
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_tube, ax_sum):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma/R_0^2$")
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

    ax_tube.set_title(r"Tube-circulation estimate")
    ax_tube.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    ax_sum.set_title(r"Total-strength-vector drift")
    ax_sum.set_ylabel(
        r"$\|\sum_p\boldsymbol{\alpha}_p-\sum_p\boldsymbol{\alpha}_{p,0}\|"
        r"\,/\,\sum_p|\boldsymbol{\alpha}_{p,0}|$"
    )

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    save_fig(
        fig,
        figs / "vortex_ring_circulation.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
