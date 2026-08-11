#!/usr/bin/env python3
"""
Energy dissipation dE/dt and −nuΩ vs t*.

Compares DNS and LES (transposed stretching) energy diagnostics parsed
from the solver log files for a single vortex ring.

Saves: figures/vortex_ring_energy.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    load_theme,
    save_fig,
    T_REF,
    P_REF,
    VARIANT_LABEL,
    VARIANT_STYLE,
    figure_size,
    mark_every,
    reference_style,
    FIGURES_DIR,
    SAMPLES_DIR,
)


def main() -> None:
    args = build_arg_parser("Energy dissipation dE/dt & -nuΩ vs t*.").parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, (ax_de, ax_nuens) = plt.subplots(1, 2, figsize=figure_size("wide_short"), sharex=True)
    legend_handles = []
    legend_labels = []

    # -- Energy diagnostics — all available variants -------------------------
    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "flow_integrals.csv"
        if not csv_path.exists():
            continue
        data = pd.read_csv(csv_path)
        times = data["time"].to_numpy()
        nuEns = data["neg_nu_enstrophy"].to_numpy()
        dedt = data["dEdt"].to_numpy()
        if times.size == 0:
            print(f"  (no energy data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {csv_path}")
        t = times / T_REF
        (line,) = ax_de.plot(
            t,
            dedt / P_REF,
            st["linestyle"],
            color=st["color"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
            label=label,
        )
        ax_nuens.plot(
            t,
            nuEns / P_REF,
            st["linestyle"],
            color=st["color"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every(),
            mew=st["markeredgewidth"],
            label=label,
        )
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_de, ax_nuens):
        ax.axhline(0.0, **reference_style())
        ax.set_xlabel(r"$t\,\Gamma / R_0^2$")
        ax.set_ylim(-0.05, 0.01)
        ax.set_xlim(0, 38)

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
        tight_rect=(0.0, 0.16, 1.0, 1.0),
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
