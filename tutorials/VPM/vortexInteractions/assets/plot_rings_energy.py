#!/usr/bin/env python3
"""
Figure 2 — Energy dissipation  dE/dt & −nuΩ vs t*

Compares energy diagnostics from leapfrog DNS, leapfrog LES, and collision LES.

Saves: figures/vortex_rings_energy.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    build_arg_parser,
    load_theme,
    parse_log,
    save_fig,
    T_REF,
    P_REF,
    CM,
)

C_DNS = "#5C3D9B"
C_LES = "#1A8C88"
C_COL = "#B85C2A"


def main():
    args = build_arg_parser("Energy dissipation dE/dt & -nuΩ vs t*.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    fig, ax = plt.subplots(figsize=(12 * CM, 7 * CM))

    runs = [
        (sol / "leapfrog_DNS" / "leapfrog_DNS.log", C_DNS, "DNS leapfrog"),
        (sol / "leapfrog_LES" / "leapfrog_LES.log", C_LES, "LES leapfrog"),
        (sol / "collide_LES" / "collide_LES.log", C_COL, "LES collision"),
    ]

    for log, color, label in runs:
        times, nuEns, dedt = parse_log(log)
        if times.size == 0:
            continue
        t = times / T_REF
        ax.plot(t, dedt / P_REF, "--", color=color, lw=1.1, label=f"{label}, $dE/dt$")
        ax.plot(
            t,
            nuEns / P_REF,
            "-",
            color=color,
            lw=1.1,
            marker="o",
            ms=3,
            markevery=3,
            label=rf"{label}, $-\nu\Omega$",
        )

    ax.axhspan(0.0, 10.0, facecolor="gray", alpha=0.15, zorder=0)
    ax.set_ylim(-0.4, 0.1)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Dissipation rate, $(dE/dt)\,T_0\,/\,(\Gamma^2 R_0)$")
    ax.legend(fontsize=10, ncol=2)
    save_fig(fig, figs / "vortex_rings_energy.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
