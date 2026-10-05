"""Final-five-revolution blade loading against the recorded-geometry BEM reference."""

import numpy as np
import pandas as pd

from openonda.plotting import centered_subplots_adjust
from openonda.results import read_csv_table
from openonda.validation import time_mean
from source.solvers.vpm.io.postprocess import chord_panel_indices, loading_window

from ._common import (
    FIGURES_DIR,
    OPERATING_WINDOW_REVOLUTIONS,
    accepted_history,
    bem_reference,
    build_arg_parser,
    load_theme,
    rotor_inputs,
    rotor_subplots,
    save_rotor_figure,
)


def shared_vlm_window(
    span, chord, *, rotation_period, revolutions=OPERATING_WINDOW_REVOLUTIONS, **kwargs
):
    return loading_window(span, chord, duration=revolutions * rotation_period, **kwargs)


def main():
    args = build_arg_parser(__doc__).parse_args()
    inputs = rotor_inputs()
    vlm = inputs.metadata["configuration"]["numerics"]["vlm"]
    surface = vlm["surfaces"][0]["name"]
    span = accepted_history(
        pd.DataFrame(read_csv_table(inputs.samples_dir / f"vlm_spanwise_{surface}.csv")),
        inputs.metadata,
    )
    chord = accepted_history(
        pd.DataFrame(read_csv_table(inputs.samples_dir / f"vlm_chordwise_{surface}.csv")),
        inputs.metadata,
    )
    expected_cadence = vlm["logging_interval_steps"] * inputs.time_step_size
    expected_chord_panels = chord_panel_indices(vlm["surfaces"][0])
    native_end_time = float(span["time"].max())
    span, chord, cutoff, end_time = shared_vlm_window(
        span,
        chord,
        rotation_period=inputs.rotation_period,
        end_time=native_end_time,
        expected_cadence=expected_cadence,
        accepted_step_size=inputs.time_step_size,
        expected_chord_panels=expected_chord_panels,
        cadence_start_time=inputs.metadata["state"]["initial_time"],
    )
    positions = chord.groupby(["step", "station_id"])[["bound_y", "bound_z"]].mean()
    positions["radius"] = np.linalg.norm(positions, axis=1)
    sampled = span.merge(
        positions[["radius"]], on=["step", "station_id"], how="left", validate="one_to_one"
    )
    fields = ["radius", "circulation_magnitude", "section_lift_coefficient_from_circulation"]
    sampled = pd.DataFrame(
        [
            time_mean(rows.time, rows[fields], cutoff, end_time)
            for _, rows in sampled.groupby("station_id")
        ],
        columns=["radius", "circulation", "cl"],
    ).sort_values("radius")
    bem = bem_reference()
    colors, _ = load_theme()
    fig, axes = rotor_subplots(2, height_cm=10, sharex=True)
    circulation_scale = inputs.freestream_speed * inputs.rotor_radius
    for axis, actual, reference in [
        (axes[0], sampled.circulation / circulation_scale, bem.circulation / circulation_scale),
        (axes[1], sampled.cl, bem.lift_coefficient),
    ]:
        axis.plot(
            sampled.radius / inputs.rotor_radius,
            actual,
            "o-",
            ms=3,
            color=colors["vpm"],
            label="VLM+VPM",
        )
        axis.plot(
            bem.normalized_radial_position, reference, "--", color=colors["reference"], label="BEM"
        )
        axis.set_xlim(inputs.hub_radius / inputs.rotor_radius, 1)
    axes[0].set_ylabel("$\\Gamma/(U_\\infty R)$")
    axes[1].set_ylabel("$c_l$")
    axes[-1].set_xlabel("Radius, $r/R$")
    handles, labels = axes[0].get_legend_handles_labels()
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
        borderaxespad=0,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    centered_subplots_adjust(fig, outer=0.129, bottom=0.23, top=0.983, hspace=0.1)
    save_rotor_figure(
        fig, FIGURES_DIR / "rotor_loading_validation.png", figure_format=args.format, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
