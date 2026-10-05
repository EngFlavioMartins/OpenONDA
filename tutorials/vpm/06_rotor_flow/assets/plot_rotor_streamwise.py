"""Plot native axial deficit and signed transverse velocities as x progresses."""

import numpy as np
import pandas as pd

from openonda.plotting import centered_subplots_adjust
from openonda.results import read_csv_table
from source.solvers.vpm.io.postprocess import profile_mean

from ..setup import STREAMWISE_STATIONS
from ._common import (
    FIGURES_DIR,
    OPERATING_WINDOW_REVOLUTIONS,
    accepted_history,
    build_arg_parser,
    load_theme,
    rotor_inputs,
    rotor_subplots,
    save_rotor_figure,
)

STREAMWISE_NAMES = tuple((f"streamwise_r{label}" for label, _ in STREAMWISE_STATIONS))


def mean_profile(data, start, end):
    """Time-average the fixed native axial line over the selected rotor interval."""
    coordinates = ["position_x", "position_y", "position_z"]
    fields = ["velocity_x", "velocity_y", "velocity_z"]
    positions, average = profile_mean(data, start, end, coordinates=coordinates, fields=fields)
    return pd.DataFrame(np.column_stack([positions, average]), columns=coordinates + fields)


def main():
    """Render one common five-revolution native window for the configured lines."""
    args = build_arg_parser(__doc__).parse_args()
    p = rotor_inputs()
    tables = [
        accepted_history(pd.DataFrame(read_csv_table(p.samples_dir / f"{name}.csv")), p.metadata)
        for name in STREAMWISE_NAMES
    ]
    end = min((table.time.max() for table in tables))
    start = end - OPERATING_WINDOW_REVOLUTIONS * p.rotation_period
    colors, _ = load_theme()
    fig, axes = rotor_subplots(3, height_cm=12.5, sharex=True)
    for table, ink, marker in zip(tables, (colors["vpm"], colors["teal"]), ("o", "s"), strict=True):
        profile = mean_profile(table, start, end)
        radial_fraction = (
            np.hypot(profile.position_y.iloc[0], profile.position_z.iloc[0]) / p.station_radius
        )
        x = profile.position_x / p.station_diameter
        for axis, field in zip(
            axes,
            (
                1.0 - profile.velocity_x / p.freestream_speed,
                profile.velocity_y / p.freestream_speed,
                profile.velocity_z / p.freestream_speed,
            ),
            strict=True,
        ):
            axis.plot(
                x,
                field,
                color=ink,
                ls="-",
                marker=marker,
                markevery=18,
                label=f"$r/R_{{\\mathrm{{d}}}}={radial_fraction:.2g}$",
            )
    for axis, label in zip(
        axes, ("$1-u_x/U_\\infty$", "$u_y/U_\\infty$", "$u_z/U_\\infty$"), strict=True
    ):
        axis.axvline(0, color="0.5", ls=":", lw=0.8)
        axis.axhline(0, color="0.7", ls=":", lw=0.6)
        axis.set_ylabel(label)
        axis.set_xlim(-0.5, 3.0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    axes[-1].set_xlabel("$x/D_{\\mathrm{d}}$ (rotor at 0)")
    centered_subplots_adjust(fig, outer=0.139, bottom=2.45 / 12.5, top=1 - 0.11 / 12.5, hspace=0.1)
    save_rotor_figure(
        fig, FIGURES_DIR / "rotor_streamwise.png", figure_format=args.format, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
