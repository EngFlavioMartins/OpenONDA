"""Plot ``delta_wing_force_cycles.png``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from openonda import plotting as _theme

from .postprocess import (
    FIGURES_DIR,
    _save_figure,
    force_history,
    last_cycles,
    motion_period,
)


def plot_force_cycles(samples_arg=None, destination=FIGURES_DIR, figure_format="png"):
    "Export the last three complete measured heave cycles, separated by wing."
    _theme.set_thesis_style()
    data = force_history(samples_arg)
    period = motion_period(data)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12.5 * _theme.CM, 8.3 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.077, bottom=0.16, top=0.81, hspace=0.27)
    for ax, surface, color, label in zip(
        axes,
        ("front_wing", "rear_wing"),
        (_theme.COLORS["teal"], _theme.COLORS["vpm"]),
        ("Front", "Rear"),
        strict=True,
    ):
        for index, (cycle, phase, tail) in enumerate(
            last_cycles(data[data.surface == surface], period)
        ):
            ax.plot(
                phase,
                tail.force_z,
                color=color,
                ls="-",
                marker=("o", "s", "D")[index],
                markevery=25,
                markersize=2.5,
                label=f"Cycle {cycle + 1}",
            )
        ax.set_ylabel("$F_z$ [N]")
        ax.set_title(label, loc="left", pad=2)
        ax.locator_params(axis="y", nbins=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
    axes[-1].set_xlabel("Cycle phase")
    _save_figure(fig, axes, destination / "delta_wing_force_cycles.png", figure_format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="both")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    args = parser.parse_args()
    plot_force_cycles(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
