#!/usr/bin/env python3
"""
Vortex-ring tube circulation and vector-sum conservation vs t*.

Saves: figures/vortex_ring_circulation.png
"""

import sys
import glob
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    load_theme,
    load_ring_circulation,
    load_vector_circulation_error,
    save_fig,
    VARIANT_LABEL,
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser(
        "Tube circulation and vector-sum conservation — DNS & LES, all stretching variants."
    ).parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, (ax_tube, ax_sum) = plt.subplots(1, 2, figsize=(12.8 * CM, 8 * CM), sharex=True)
    legend_handles = []
    legend_labels = []

    for variant, st in VARIANT_STYLE.items():
        h5 = sorted(glob.glob(str(sol / variant / f"vpm_{variant}_*.h5")))
        t, c = load_ring_circulation(h5)
        if t.size == 0:
            continue
        t_sum, sum_err = load_vector_circulation_error(h5)
        ls = "--" if variant.startswith("DNS") else "-"
        label = VARIANT_LABEL[variant]
        (line,) = ax_tube.plot(
            t,
            c,
            ls,
            color=st["color"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=3,
            mew=0.4,
            label=label,
        )
        if t_sum.size:
            ax_sum.semilogy(
                t_sum,
                sum_err.clip(min=1e-12),
                ls,
                color=st["color"],
                lw=1.1,
                marker=st["marker"],
                ms=3,
                markevery=3,
                mew=0.4,
                label=label,
            )
        legend_handles.append(line)
        legend_labels.append(label)

    ax_tube.axhline(
        1.0,
        color="gray",
        ls="--",
        lw=1.0,
    )
    ax_sum.axhline(1e-4, color="gray", ls="--", lw=1.0)

    for ax in (ax_tube, ax_sum):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
        ax.set_xlim(0, 38)

    ax_tube.set_title(r"Tube circulation")
    ax_tube.set_ylabel(r"$\Gamma_{\rm tube}/\Gamma_{\rm tube,0}$")
    ax_tube.set_ylim(0.5, 1.5)
    ax_sum.set_title(r"Vector-sum conservation")
    ax_sum.set_ylabel(r"$\|\Sigma\alpha-\Sigma\alpha_0\|\,/\,\Sigma|\alpha|_0$")
    ax_sum.set_ylim(1e-8, 1.0)

    fig.legend(
        legend_handles,
        legend_labels,
        fontsize=10,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    save_fig(
        fig,
        figs / "vortex_ring_circulation.png",
        dpi=args.dpi,
        tight_rect=(0.0, 0.16, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
