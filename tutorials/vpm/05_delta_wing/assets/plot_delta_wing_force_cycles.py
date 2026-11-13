"""Plot ``delta_wing_force_cycles.png``."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as _theme

from assets.postprocess import (
    FIGURES_DIR,
    _save_figure,
    force_history,
    last_cycles,
    lift_and_thrust,
    motion_period,
)


def plot_force_cycles(samples_arg=None, destination=FIGURES_DIR, figure_format="png", count=4):
    "Export the heave, lift and thrust of both wings, phase-averaged over the final cycles."
    _theme.set_thesis_style()
    data = lift_and_thrust(force_history(samples_arg))
    period = motion_period(data)
    phase = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.5 * _theme.CM, 10.5 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.12, bottom=0.13, top=0.89, hspace=0.2)
    for surface, color, label in (
        ("front_wing", _theme.COLORS["teal"], "Front"),
        ("rear_wing", _theme.COLORS["vpm"], "Rear"),
    ):
        cycles = list(last_cycles(data[data.surface == surface], period, count))
        for ax, column in zip(
            axes, ("centroid_z", "lift_coefficient", "thrust_coefficient"), strict=True
        ):
            values = np.array(
                [np.interp(phase, fraction, rows[column]) for _, fraction, rows in cycles]
            )
            ax.fill_between(phase, values.min(axis=0), values.max(axis=0), color=color, alpha=0.3)
            ax.plot(phase, values.mean(axis=0), color=color, label=label)
            print(f"{label} {column}: cycle mean {values.mean():.3g}")
    print(f"Cycles {cycles[0][0] + 1}-{cycles[-1][0] + 1}")
    for ax, label in zip(axes, ("$z_c$ [m]", "$C_L$", "$C_T$"), strict=True):
        ax.set_ylabel(label)
        ax.axhline(0, color=_theme.COLORS["reference"], lw=0.6, linestyle="--")
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xlabel("Cycle phase, $t\\,f$")
    axes[-1].set_xlim(0.0, 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2)
    _theme.fit_thesis_y_label_margins(fig, axes)
    _save_figure(fig, destination / "delta_wing_force_cycles.png", figure_format)


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
    plot_force_cycles(args.samples, FIGURES_DIR, args.format)


if __name__ == "__main__":
    main()
