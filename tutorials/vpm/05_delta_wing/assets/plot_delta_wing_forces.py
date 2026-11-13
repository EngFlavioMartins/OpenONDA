"""Plot ``delta_wing_forces.png``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from openonda import plotting as _theme

from assets.postprocess import (
    FIGURES_DIR,
    _save_figure,
    force_history,
    lift_and_thrust,
    motion_period,
)


def plot_forces(samples_arg=None, destination=FIGURES_DIR, figure_format="png"):
    "Export the lift, thrust and input-power histories of both wings."
    _theme.set_thesis_style()
    data = lift_and_thrust(force_history(samples_arg))
    data = data.assign(cycles=data.time / motion_period(data), input_power=-data.power)
    columns = ("lift_coefficient", "thrust_coefficient", "input_power")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.5 * _theme.CM, 10.5 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.12, bottom=0.13, top=0.89, hspace=0.2)
    for surface, color, label in (
        ("front_wing", _theme.COLORS["teal"], "Front"),
        ("rear_wing", _theme.COLORS["vpm"], "Rear"),
    ):
        rows = data[data.surface == surface]
        for ax, column in zip(axes, columns, strict=True):
            ax.plot(rows.cycles, rows[column], color=color, label=label)
    settled = data[data.cycles >= 1.0]
    for ax, column, label in zip(
        axes, columns, ("$C_L$", "$C_T$", "$P_{\\mathrm{in}}$ [W]"), strict=True
    ):
        low, high = settled[column].min(), settled[column].max()
        ax.set_ylim(low - 0.1 * (high - low), high + 0.1 * (high - low))
        ax.set_ylabel(label)
        ax.axhline(0, color=_theme.COLORS["reference"], lw=0.6, linestyle="--")
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xlabel("Heave cycles, $t\\,f$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
    _theme.fit_thesis_y_label_margins(fig, axes)
    _save_figure(fig, destination / "delta_wing_forces.png", figure_format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=_theme.FORMAT_CHOICES, default="png")
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory; repeat in sparse-to-dense order for a continuation",
    )
    args = parser.parse_args()
    plot_forces(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
