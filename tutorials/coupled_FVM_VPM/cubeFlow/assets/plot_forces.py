#!/usr/bin/env python3
"""Cube force history: drag/lift coefficients and IBM no-slip quality."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

    cd, cl = forces["Cd"], forces["Cl"]
    ax_c.plot(t_star, cd, color=COLORS["cd"], marker="o", ms=3, label=r"$C_D$")
    ax_c.plot(t_star, cl, color=COLORS["cl"], marker="s", ms=3, label=r"$C_L$")
    ax_c.axhline(0.0, color=COLORS["box"], lw=0.8, zorder=0)
    ax_c.set(ylabel="force coefficient", title="Cube force history (immersed-boundary)")
    ax_c.legend(loc="upper right", ncol=2)

    # The impulsive start produces a large one-step C_D spike (the body appears
    # in the flow instantaneously); focus the axis on the settled range and
    # note the peak rather than letting it flatten everything else.
    settled = cd[2:] if cd.size > 3 else cd
    top = max(float(np.max(settled)), float(np.max(np.abs(cl)))) * 1.4 + 0.1
    if float(cd[0]) > top:
        ax_c.set_ylim(min(0.0, float(np.min(cl)) * 1.4), top)
        ax_c.annotate(
            rf"start peak $C_D\approx{cd[0]:.0f}$ (off-scale)",
            xy=(t_star[0], top),
            xytext=(0.05, 0.88),
            textcoords="axes fraction",
            fontsize=8,
            color=COLORS["accent"],
        )

    if "Fpx" in forces:
        # Wall-patch data: pressure vs viscous drag split (q recovered from
        # the logged total force and coefficient).
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(forces["Cd"] != 0.0, forces["Ftx"] / forces["Cd"], np.nan)
        ax_s.plot(
            t_star,
            forces["Fpx"] / q,
            color=COLORS["fvm"],
            marker="o",
            ms=3,
            label=r"$C_{D,\mathrm{pressure}}$",
        )
        ax_s.plot(
            t_star,
            forces["Fvx"] / q,
            color=COLORS["vpm"],
            marker="s",
            ms=3,
            label=r"$C_{D,\mathrm{viscous}}$",
        )
        ax_s.set(
            xlabel=r"$t\,U_\infty / D$",
            ylabel="drag contribution",
            title="Pressure / viscous drag split (wall patch)",
        )
        ax_s.legend(loc="upper right", ncol=2)
        lim = ax_c.get_ylim()
        ax_s.set_ylim(min(0.0, lim[0]), lim[1])
    else:
        # Immersed-boundary data: marker-slip quality monitor.
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
