#!/usr/bin/env python3
"""Resolved and modeled viscous energy rates for the vortex-ring cases.

The resolved rate is a backward difference of the reconstructed-field kinetic
energy.  The modeled viscous rate is evaluated directly with each particle's
effective viscosity, including the Smagorinsky contribution in the LES case.

Saves: figures/vortex_ring_energy.png
"""

import matplotlib.pyplot as plt
import pandas as pd

from tutorials.vpm.vortex_ring.assets.ring_metrics import (
    FIGURES_DIR,
    P_REF,
    SAMPLES_DIR,
    REFERENCE_TIME,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
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
        "Resolved and modeled viscous energy rates versus normalized time."
    ).parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, (ax_de, ax_nuens) = plt.subplots(1, 2, figsize=figure_size("wide_short"), sharex=True)
    fig.subplots_adjust(wspace=0.50, left=0.14, right=0.87, top=0.92, bottom=0.29)
    legend_handles = []
    legend_labels = []
    n_skip = 14  # plot every n-th marker

    plotted_values = []
    maximum_time = 0.0

    # -- Energy diagnostics — all available variants -------------------------
    for variant in plot_variants():
        st = VARIANT_STYLE[variant]
        csv_path = SAMPLES_DIR / variant / "flow_integrals.csv"
        if not csv_path.exists():
            continue
        data = pd.read_csv(csv_path)
        keep = data["time"].to_numpy() > 0.0
        times = data["time"].to_numpy()[keep]
        modeled_dissipation = -data["viscous_kinetic_energy_rate"].to_numpy()[keep] / P_REF
        resolved_dissipation = -data["kinetic_energy_rate"].to_numpy()[keep] / P_REF
        if times.size == 0:
            print(f"  (no energy data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {csv_path}")
        t = times / REFERENCE_TIME
        resolved_valid = resolved_dissipation > 0.0
        modeled_valid = modeled_dissipation > 0.0
        (line,) = ax_de.plot(
            t[resolved_valid],
            resolved_dissipation[resolved_valid],
            label=label,
            **line_style(st, n_skip),
        )
        ax_nuens.plot(
            t[modeled_valid],
            modeled_dissipation[modeled_valid],
            label=label,
            **line_style(st, n_skip),
        )
        plotted_values.extend(
            (resolved_dissipation[resolved_valid], modeled_dissipation[modeled_valid])
        )
        maximum_time = max(maximum_time, float(t[-1]))
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_de, ax_nuens):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma/R_0^2$")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.01 * maximum_time)
    finite_values = [values for values in plotted_values if values.size]
    if finite_values:
        lower = min(float(values.min()) for values in finite_values)
        upper = max(float(values.max()) for values in finite_values)
        for ax in (ax_de, ax_nuens):
            ax.set_ylim(0.8 * lower, 1.25 * upper)

    ax_de.set_title(r"Resolved dissipation rate")
    ax_de.set_ylabel(r"$-(\Delta E_h/\Delta t)\,/\,(\Gamma^3/R_0)$")
    ax_nuens.set_title(r"Modeled viscous dissipation")
    ax_nuens.set_ylabel(r"$-(\mathrm{d}E_h/\mathrm{d}t)_{\nu}\,/\,(\Gamma^3/R_0)$")
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    save_fig(
        fig,
        figs / "vortex_ring_energy.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
