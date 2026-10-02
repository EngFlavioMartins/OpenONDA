#!/usr/bin/env python3
"""Plot native axial deficit and signed transverse velocities as x progresses."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from openonda.plotting import centered_subplots_adjust

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

STREAMWISE_NAMES = tuple(f"streamwise_r{label}" for label, _ in STREAMWISE_STATIONS)


def mean_profile(data, start, end):
    """Time-integrate a complete fixed native line over [start, end] in seconds.

    Linear interpolation is confined to the two window boundaries. Every
    snapshot must contain the same positions exactly once. Missing/nonfinite
    data or an incomplete time window raise ValueError; no field is synthesized.
    """
    columns = ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
    if end <= start or not np.isfinite([start, end]).all():
        raise ValueError("streamwise averaging requires a finite positive time window")
    if data.empty or not np.isfinite(data[["time", *columns]].to_numpy()).all():
        raise ValueError("streamwise samples must be nonempty and finite")
    times = np.sort(data.time.unique())
    if times[0] > start + 1e-10 or times[-1] < end - 1e-10:
        raise ValueError("streamwise samples do not cover the complete averaging window")
    frames = [data[data.time == t].sort_values("position_x") for t in times]
    positions = frames[0][columns[:3]].to_numpy()
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError("duplicate points in native streamwise snapshot")
    for frame in frames:
        if not np.array_equal(frame[columns[:3]].to_numpy(), positions):
            raise ValueError("streamwise snapshot has missing, duplicated or moving points")
    values = np.stack([frame[columns[3:]].to_numpy() for frame in frames])
    knots = np.r_[start, times[(times > start) & (times < end)], end]
    sampled = np.empty((len(knots), len(positions), 3))
    for point in range(len(positions)):
        for component in range(3):
            sampled[:, point, component] = np.interp(knots, times, values[:, point, component])
    average = trapezoid(sampled, x=knots, axis=0) / (end - start)
    return pd.DataFrame(np.column_stack([positions, average]), columns=columns)


def main():
    """Render one common five-revolution native window for the configured lines."""
    args = build_arg_parser(__doc__).parse_args()
    p = rotor_inputs()
    tables = [
        accepted_history(pd.read_csv(p.samples_dir / f"{name}.csv"), p.metadata)
        for name in STREAMWISE_NAMES
    ]
    end = min(table.time.max() for table in tables)
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
                label=rf"$r/R_{{\mathrm{{d}}}}={radial_fraction:.2g}$",
            )
    for axis, label in zip(
        axes, (r"$1-u_x/U_\infty$", r"$u_y/U_\infty$", r"$u_z/U_\infty$"), strict=True
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
    axes[-1].set_xlabel(r"$x/D_{\mathrm{d}}$ (rotor at 0)")
    centered_subplots_adjust(fig, outer=0.139, bottom=2.45 / 12.5, top=1 - 0.11 / 12.5, hspace=0.10)
    save_rotor_figure(
        fig, FIGURES_DIR / "rotor_streamwise.png", figure_format=args.format, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
