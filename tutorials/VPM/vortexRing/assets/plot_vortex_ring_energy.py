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
    VARIANT_STYLE,
    CM,
)


def main() -> None:
    args = build_arg_parser("Energy dissipation dE/dt & -nuΩ vs t*.").parse_args()
    sol = Path(args.solution_dir)
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()

    dns_st = VARIANT_STYLE["DNS_transposed"]
    les_st = VARIANT_STYLE["LES_rvpm"]

    dns_log = sol / "DNS_transposed" / "DNS_transposed.log"
    les_log = sol / "LES_rvpm" / "LES_rvpm.log"

    fig, ax = plt.subplots(figsize=(12 * CM, 7 * CM))

    for log, st, ls, label in [
        (dns_log, dns_st, "--", "DNS"),
        (les_log, les_st, "-", "In-house LES"),
    ]:
        times, nuEns, dedt = parse_log(log)
        if times.size == 0:
            continue
        t = times / T_REF
        ax.plot(t, dedt / P_REF, ls, color=st["color"], lw=1.1, label=f"{label}, $dE/dt$")
        ax.plot(
            t,
            nuEns / P_REF,
            ls,
            color=st["color"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=3,
            label=rf"{label}, $-\nu\,\varepsilon$",
        )

    ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel(r"Normalized time, $t\,\Gamma / R_0^2$")
    ax.set_ylabel(r"Dissipation rate, $(dE/dt)\,T_0\,/\,(\Gamma^2 R_0)$")
    ax.set_ylim(-0.06, 0.02)
    ax.set_xlim(0, 38)
    ax.legend(fontsize=10, ncol=2)
    save_fig(fig, figs / "vortex_ring_energy.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
