#!/usr/bin/env python3
"""
Energy dissipation dE/dt and −kinematic_viscosityΩ vs t*.

Compares DNS and LES (transposed stretching) energy diagnostics parsed
from the solver log files for a single vortex ring.

Saves: figures/vortex_ring_energy.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ring_metrics import (
    FIGURES_DIR,
    P_REF,
    SAMPLES_DIR,
    REFERENCE_TIME,
    VARIANT_LABEL,
    VARIANT_STYLE,
    build_arg_parser,
    figure_size,
    load_theme,
    save_fig,
)

ZOOM_END = 5
ZOOM_BOX = (0.42, 0.12, 0.50, 0.35)
ZOOM_MARKER_SIZE = 2.0
ZOOM_FLOOR = 1e-3


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
    inset.set_yscale("symlog", linthresh=ZOOM_FLOOR)
    inset.set_ylim(*ylim)
    return inset


def main() -> None:
    args = build_arg_parser("Energy dissipation dE/dt & -kinematic_viscosityΩ vs t*.").parse_args()
    figs = FIGURES_DIR
    figs.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()

    fig, (ax_de, ax_nuens) = plt.subplots(1, 2, figsize=figure_size("wide_short"), sharex=True)
    fig.subplots_adjust(wspace=0.50, left=0.14, right=0.87, top=0.92, bottom=0.29)
    legend_handles = []
    legend_labels = []
    plot_end = 185
    n_skip = 14  # plot every n-th marker

    zoom_de = zoom_axes(ax_de, ylim=(-2e-1, 5e-2))

    # -- Energy diagnostics — all available variants -------------------------
    for variant, st in VARIANT_STYLE.items():
        csv_path = SAMPLES_DIR / variant / "flow_integrals.csv"
        if not csv_path.exists():
            continue
        data = pd.read_csv(csv_path)
        keep = data["time"].to_numpy() > 0.0
        times = data["time"].to_numpy()[keep]
        viscous_kinetic_energy_rate = data["viscous_kinetic_energy_rate"].to_numpy()[keep]
        dedt = data["kinetic_energy_rate"].to_numpy()[keep]
        if times.size == 0:
            print(f"  (no energy data for {variant})")
            continue
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {csv_path}")
        t = times / REFERENCE_TIME
        (line,) = ax_de.plot(t, dedt / P_REF, label=label, **line_style(st, n_skip))
        ax_nuens.plot(t, viscous_kinetic_energy_rate / P_REF, label=label, **line_style(st, n_skip))
        zoom_de.plot(t, dedt / P_REF, **line_style(st, 1, ZOOM_MARKER_SIZE))
        legend_handles.append(line)
        legend_labels.append(label)

    for ax in (ax_de, ax_nuens):
        ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
        ax.set_yscale("symlog", linthresh=ZOOM_FLOOR)
        ax.set_ylim(-5e-1, -5e-4)

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
