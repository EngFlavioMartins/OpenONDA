#!/usr/bin/env python3
"""Publication-style cube force history from sampled wall loads."""

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_DPI = 400
FORCE_MARKERS = 20


def _plot_force(ax, time, values, *, color, marker, label, linestyle="-") -> None:
    ax.plot(
        time,
        values,
        color=color,
        ls=linestyle,
        marker=marker,
        ms=3,
        markevery=max(1, len(time) // FORCE_MARKERS),
        label=label,
    )


def _drag_split(forces: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(forces["Cd"] != 0.0, forces["Ftx"] / forces["Cd"], np.nan)
    return forces["Fpx"] / scale, forces["Fvx"] / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()

    coupled = util.load_forces("fvm")
    reference = util.load_forces("reference")
    if coupled is None or reference is None:
        raise SystemExit("No coupled/reference forces_history.csv found in samples/.")

    consts = util.run_constants()
    t_coupled = coupled["time"] * consts["U_inf"] / consts["D"]
    t_reference = reference["time"] * consts["U_inf"] / consts["D"]
    fig, (ax_force, ax_split) = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)

    _plot_force(
        ax_force,
        t_coupled,
        coupled["Cd"],
        color=util.COLORS["cd"],
        marker="o",
        label=r"Hybrid $C_D$",
    )
    _plot_force(
        ax_force,
        t_coupled,
        coupled["Cl"],
        color=util.COLORS["cl"],
        marker="s",
        label=r"Hybrid $C_L$",
    )
    ax_force.plot(
        t_reference,
        reference["Cd"],
        color=util.COLORS["reference"],
        ls="-.",
        label=r"Reference $C_D$",
    )
    ax_force.axhline(0.0, color=util.COLORS["box"], lw=0.8, zorder=0)
    ax_force.set(ylabel="force coefficient", title="Cube force history")
    ax_force.legend(loc="upper right", ncol=2)

    pressure_c, viscous_c = _drag_split(coupled)
    pressure_r, viscous_r = _drag_split(reference)
    _plot_force(
        ax_split,
        t_coupled,
        pressure_c,
        color=util.COLORS["fvm"],
        marker="o",
        label=r"Hybrid $C_{D,p}$",
    )
    _plot_force(
        ax_split,
        t_coupled,
        viscous_c,
        color=util.COLORS["vpm"],
        marker="s",
        label=r"Hybrid $C_{D,\nu}$",
    )
    ax_split.plot(
        t_reference,
        pressure_r,
        color=util.COLORS["fvm"],
        ls="-.",
        label=r"Reference $C_{D,p}$",
    )
    ax_split.plot(
        t_reference,
        viscous_r,
        color=util.COLORS["vpm"],
        ls="-.",
        label=r"Reference $C_{D,\nu}$",
    )
    ax_split.set(
        xlabel=r"$t U_\infty/D$",
        ylabel="drag contribution",
        title="Pressure / viscous drag split",
    )
    ax_split.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    util.save(fig, "forces_history", args.format, FIGURE_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
