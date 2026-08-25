#!/usr/bin/env python3
"""
Energy dissipation dE/dt and −kinematic_viscosityΩ vs t*.

Compares DNS and LES (transposed stretching) energy diagnostics parsed
from the solver log files for a single vortex ring.

Saves: figures/vortex_ring_energy.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import (
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_theme,
    save_fig,
)
from ring_metrics import FIGURES_DIR, P_REF, SAMPLES_DIR, REFERENCE_TIME


def main() -> None:
    args = build_arg_parser("Energy dissipation dE/dt & -kinematic_viscosityΩ vs t*.").parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()

    fig, (ax_de, ax_nuens) = plt.subplots(1, 2, figsize=figure_size("wide_short"), sharex=True)
    fig.subplots_adjust(wspace=0.37, left=0.13, right=0.98, top=0.92, bottom=0.29)
    legend_handles = []
    legend_labels = []
    plot_end = 185
    n_skip = 14  # plot every n-th marker

    # -- Energy diagnostics — all available variants -------------------------
    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "flow_integrals.csv"
        if not csv_path.exists():
            continue
        data = pd.read_csv(csv_path)
        times = data["time"].to_numpy()
        viscous_kinetic_energy_rate = data["viscous_kinetic_energy_rate"].to_numpy()
        dedt = data["kinetic_energy_rate"].to_numpy()
        if times.size == 0:
            print(f"  (no energy data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {csv_path}")
        t = times / REFERENCE_TIME
        (line,) = ax_de.plot(
            t,
            dedt / P_REF,
            linestyle=st["linestyle"],
            color=st["color"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=n_skip,
            mew=st["markeredgewidth"],
            label=label,
        )
        ax_nuens.plot(
            t,
            viscous_kinetic_energy_rate / P_REF,
            linestyle=st["linestyle"],
            color=st["color"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=n_skip,
            mew=st["markeredgewidth"],
            label=label,
        )
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_de, ax_nuens):
        ax.set_xlabel(r"$t\,\Gamma / R_0^2$")
        ax.set_ylim(-0.05, 0.01)
        ax.set_xlim(0, plot_end)
        ax.axhspan(0.0, 0.01, color=colors["background_light"], linewidth=0, zorder=0)

    ax_de.set_title(r"Energy rate versus time")
    ax_de.set_ylabel(r"$(dE/dt)\,T_0\,/\,(\Gamma^2 R_0)$")
    ax_nuens.set_title(r"Viscous dissipation versus time")
    ax_nuens.set_ylabel(r"$(-\nu\varepsilon)\,T_0\,/\,(\Gamma^2 R_0)$")
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=3,
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
