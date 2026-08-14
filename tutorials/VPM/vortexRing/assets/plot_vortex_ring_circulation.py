#!/usr/bin/env python3
"""
Vortex-ring tube circulation and vector-sum conservation vs t*.

Saves: figures/vortex_ring_circulation.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_theme,
    save_fig,
)
from ring_metrics import (
    FIGURES_DIR,
    SAMPLES_DIR,
    load_sampled_ring_circulation,
    load_sampled_vector_circulation_error,
)


def main() -> None:
    args = build_arg_parser(
        "Tube circulation and vector-sum conservation — DNS & LES, all stretching variants."
    ).parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()

    fig, (ax_tube, ax_sum) = plt.subplots(1, 2, figsize=figure_size("single_tall"), sharex=True)
    fig.subplots_adjust(wspace=0.32, hspace=0.10, left=0.10, right=0.98, top=0.92, bottom=0.30)
    legend_handles = []
    legend_labels = []

    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "ring_diagnostics.csv"
        t, c = load_sampled_ring_circulation(csv_path)
        if t.size == 0:
            continue
        t_sum, sum_err = load_sampled_vector_circulation_error(csv_path)
        label = VARIANT_LABEL[variant]
        (line,) = ax_tube.plot(
            t,
            c,
            linestyle="None",
            color=st["color"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=2,
            mew=st["markeredgewidth"],
            label=label,
        )
        if t_sum.size:
            ax_sum.semilogy(
                t_sum,
                sum_err.clip(min=1e-12),
                linestyle="None",
                color=st["color"],
                marker=st["marker"],
                ms=st["markersize"],
                markevery=2,
                mew=st["markeredgewidth"],
                label=label,
            )
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_tube, ax_sum):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
        ax.set_xlim(0, 38)

    ax_tube.set_title(r"Tube circulation")
    ax_tube.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    ax_tube.set_ylim(0.5, 1.5)
    ax_sum.set_title(r"Vector-sum conservation")
    ax_sum.set_ylabel(r"$\|\Sigma\alpha-\Sigma\alpha_0\|\,/\,\Sigma|\alpha|_0$")
    ax_sum.set_ylim(1e-8, 1.0)
    ax_sum.axhspan(1e-4, 1.0, color=colors["background_light"], linewidth=0, zorder=0)

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
