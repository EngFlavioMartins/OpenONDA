#!/usr/bin/env python3
"""Cube force history: drag/lift coefficients and IBM no-slip quality."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plotutil import COLORS, apply_style, load_forces, parse_args, run_constants, save


def main() -> int:
    args = parse_args()
    apply_style()
    forces = load_forces()
    if not forces:
        print("  no ibm_forces_history.csv — run the case first; skipping force plot.")
        return 0

    const = run_constants()
    t_star = forces["time"] * const["U_inf"] / const["D"]

    fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)

    ax_c.plot(t_star, forces["Cd"], color=COLORS["cd"], marker="o", ms=3, label=r"$C_D$")
    ax_c.plot(t_star, forces["Cl"], color=COLORS["cl"], marker="s", ms=3, label=r"$C_L$")
    ax_c.axhline(0.0, color=COLORS["box"], lw=0.8, zorder=0)
    ax_c.set(ylabel="force coefficient", title="Cube force history (immersed-boundary)")
    ax_c.legend(loc="upper right", ncol=2)

    ax_s.plot(t_star, forces["slip"], color=COLORS["accent"], marker=".", ms=4)
    ax_s.set(
        xlabel=r"$t\,U_\infty / D$",
        ylabel="marker slip",
        title="No-slip enforcement quality (lower is better)",
    )
    ax_s.set_ylim(bottom=0.0)

    fig.tight_layout()
    save(fig, "forces_history", args.format, args.dpi)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
