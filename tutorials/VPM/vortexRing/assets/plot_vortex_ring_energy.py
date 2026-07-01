#!/usr/bin/env python3
"""
Energy dissipation dE/dt and −nuΩ vs t*.

Compares DNS and LES (transposed stretching) energy diagnostics parsed
from the solver log files for a single vortex ring.

Saves: figures/vortex_ring_energy.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    load_theme,
    parse_log,
    save_fig,
    T_REF,
    P_REF,
    VARIANT_LABEL,
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser("Energy dissipation dE/dt & -nuΩ vs t*.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, (ax_de, ax_nuens) = plt.subplots(1, 2, figsize=(17.2 * CM, 8 * CM), sharex=True)
    legend_handles = []
    legend_labels = []

    # -- Energy diagnostics — all available variants -------------------------
    for variant, st in VARIANT_STYLE.items():
        log = sol / variant / f"{variant}.log"
        if not log.exists():
            continue
        times, nuEns, dedt = parse_log(log)
        if times.size == 0:
            print(f"  (no energy data for {variant})")
            continue
        ls = "--" if variant.startswith("DNS") else "-"
        label = VARIANT_LABEL[variant]
        print(f"  {variant}: {log}")
        t = times / T_REF
        (line,) = ax_de.plot(
            t,
            dedt / P_REF,
            ls,
            color=st["color"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=3,
            mew=0.4,
            label=label,
        )
        ax_nuens.plot(
            t,
            nuEns / P_REF,
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

    for ax in (ax_de, ax_nuens):
        ax.axhline(0.0, color="gray", ls="--", lw=1.0)
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
        fontsize=10,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    save_fig(fig, figs / "vortex_ring_energy.png", dpi=args.dpi, tight_rect=(0.0, 0.16, 1.0, 1.0))


if __name__ == "__main__":
    main()
