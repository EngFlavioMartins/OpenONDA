#!/usr/bin/env python3
"""
Vortex-ring tube circulation and vector-sum conservation versus t Gamma/R0^2.

Saves: figures/vortex_ring_circulation.png
"""

import matplotlib.pyplot as plt

from ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_sampled_ring_circulation,
    load_sampled_vector_circulation_error,
    load_theme,
    save_fig,
)

ZOOM_END = 3.2
ZOOM_BOX = (0.54, 0.64, 0.40, 0.32)
ZOOM_MARKER_SIZE = 2.0


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


def zoom_axes(ax, ylim: tuple[float, float]):
    """Attach an early-time zoom inset to the top-right corner of ``ax``."""
    inset = ax.inset_axes(ZOOM_BOX)
    inset.set_xlim(0.0, ZOOM_END)
    inset.set_ylim(*ylim)
    return inset


def main() -> None:
    args = build_arg_parser(
        "Tube-circulation estimate and vector-strength drift for all vortex-ring variants."
    ).parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, (ax_tube, ax_sum) = plt.subplots(1, 2, figsize=figure_size("single_tall"), sharex=True)
    fig.subplots_adjust(wspace=0.35, hspace=0.10, left=0.09, right=0.91, top=0.92, bottom=0.30)
    legend_handles = []
    legend_labels = []

    n_skip = 14  # plot every n-th marker

    ax_sum.set_yscale("log")
    zoom_tube = zoom_axes(ax_tube, ylim=(0.98, 1.80))
    zoom_sum = zoom_axes(ax_sum, ylim=(1e-9, 1e-1))
    zoom_sum.set_yscale("log")

    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "ring_diagnostics.csv"
        t, c = load_sampled_ring_circulation(csv_path)
        if t.size == 0:
            continue
        t_sum, sum_err = load_sampled_vector_circulation_error(csv_path)
        label = VARIANT_LABEL[variant]
        (line,) = ax_tube.plot(t, c, label=label, **line_style(st, n_skip))
        zoom_tube.plot(t, c, **line_style(st, 1, ZOOM_MARKER_SIZE))
        if t_sum.size:
            ax_sum.semilogy(t_sum, sum_err.clip(min=1e-12), label=label, **line_style(st, n_skip))
            zoom_sum.semilogy(t_sum, sum_err.clip(min=1e-12), **line_style(st, 1, ZOOM_MARKER_SIZE))
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_tube, ax_sum):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma/R_0^2$")

    ax_tube.set_title(r"Tube-circulation estimate")
    ax_tube.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    ax_tube.set_ylim(1.0, 1.6)
    ax_sum.set_title(r"Total-strength-vector drift")
    ax_sum.set_ylabel(
        r"$\|\sum_p\boldsymbol{\alpha}_p-\sum_p\boldsymbol{\alpha}_{p,0}\|"
        r"\,/\,\sum_p|\boldsymbol{\alpha}_{p,0}|$"
    )
    ax_sum.set_ylim(1e-9, 1e-1)

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=3,
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
