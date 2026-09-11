#!/usr/bin/env python3
"""Final-five-revolution blade loading against the recorded-geometry BEM reference."""

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ._common import (
    FIGURES_DIR,
    OPERATING_WINDOW_REVOLUTIONS,
    bem_reference,
    build_arg_parser,
    load_theme,
    rotor_inputs,
)


def _native_vlm_clock(data, label):
    """Return one strictly increasing accepted-step/time clock per VLM history."""
    required = {"step", "time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} history lacks required clock columns: {sorted(missing)}")
    clock = data[["step", "time"]].drop_duplicates()
    if clock["step"].duplicated().any() or clock["time"].duplicated().any():
        raise ValueError(f"{label} history has duplicate or conflicting clock rows")
    steps = clock["step"].to_numpy()
    times = clock["time"].to_numpy(dtype=float)
    if len(clock) < 2 or not np.isfinite(times).all():
        raise ValueError(f"{label} history has an incomplete or non-finite clock")
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{label} history clock is not strictly increasing")
    return steps, times


def _native_vlm_station_keys(
    data,
    label,
    *,
    panel_column=None,
    expected_panel_indices=None,
):
    """Validate native station keys, retaining chordwise panel identity.

    Spanwise histories have one row per ``(step, station_id)``.  Chordwise
    histories intentionally have several rows for that same station, so their
    uniqueness contract is ``(step, station_id, chord_index)``.  The grouped
    station keys returned for the span/chord merge are only produced after
    every station has the complete panel set.
    """
    required = {"step", "station_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} history lacks station keys: {sorted(missing)}")
    station_columns = ["step", "station_id"]
    if panel_column is None:
        keys = data[station_columns]
        if keys.duplicated().any():
            raise ValueError(f"{label} history has duplicate step/station keys")
    else:
        if panel_column not in data.columns:
            raise ValueError(f"{label} history lacks panel key: {panel_column!r}")
        panel_columns = station_columns + [panel_column]
        panel_keys = data[panel_columns]
        if panel_keys[panel_column].isna().any():
            raise ValueError(f"{label} history has non-finite chord-panel keys")
        if panel_keys.duplicated().any():
            raise ValueError(f"{label} history has duplicate step/station/chord-panel keys")
        observed_panels = frozenset(panel_keys[panel_column].unique())
        expected_panels = (
            observed_panels if expected_panel_indices is None else frozenset(expected_panel_indices)
        )
        if not expected_panels:
            raise ValueError(f"{label} history has no chord-panel keys")
        panel_sets = panel_keys.groupby(station_columns, sort=False)[panel_column].agg(
            lambda values: frozenset(values)
        )
        if any(panels != expected_panels for panels in panel_sets):
            raise ValueError(f"{label} history has incomplete chord panels")
        keys = panel_keys[station_columns].drop_duplicates()
    return set(keys.itertuples(index=False, name=None))


def _native_vlm_logging_cadence(metadata, time_step_size):
    """Return the mandatory owner-step force-history cadence."""
    try:
        raw_interval = metadata["configuration"]["numerics"]["vlm"]["logging_interval_steps"]
        interval_steps = int(raw_interval)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("native VLM logging_interval_steps is invalid") from exc
    if isinstance(raw_interval, bool) or interval_steps < 1 or interval_steps != raw_interval:
        raise ValueError("native VLM logging_interval_steps must be a positive integer")
    if interval_steps != 1:
        raise ValueError(
            "coupled rotor VLM force/loading history must use every accepted owner step"
        )
    time_step_size = float(time_step_size)
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("native VLM time step size must be finite and positive")
    return interval_steps * time_step_size


def _native_vlm_expected_chord_panels(surface):
    """Read a uniform chord-panel contract from the native VLM surface mesh."""
    segments = [segment for wing in surface["geometry"]["wings"] for segment in wing["segments"]]
    counts = {int(segment["n_chordwise_panels"]) for segment in segments}
    if len(counts) != 1:
        return None
    count = counts.pop()
    if count < 1:
        raise ValueError("native VLM surface has no chordwise panels")
    return tuple(range(count))


def shared_vlm_window(
    span,
    chord,
    *,
    rotation_period,
    revolutions=OPERATING_WINDOW_REVOLUTIONS,
    end_time=None,
    expected_cadence=None,
    accepted_step_size=None,
    expected_chord_panels=None,
):
    """Select one shared final window after reconciling span/chord clocks."""
    if revolutions <= 0:
        raise ValueError("revolutions must be positive")
    span_steps, span_times = _native_vlm_clock(span, "spanwise")
    chord_steps, chord_times = _native_vlm_clock(chord, "chordwise")
    if not np.array_equal(span_steps, chord_steps) or not np.allclose(
        span_times, chord_times, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("spanwise and chordwise VLM clocks do not match")
    span_keys = _native_vlm_station_keys(span, "spanwise")
    chord_keys = _native_vlm_station_keys(
        chord,
        "chordwise",
        panel_column="chord_index",
        expected_panel_indices=expected_chord_panels,
    )
    if span_keys != chord_keys:
        raise ValueError("spanwise and chordwise step/station keys do not match")
    if expected_cadence is not None:
        if expected_cadence <= 0.0:
            raise ValueError("expected_cadence must be positive")
        accepted_step_size = 0.0 if accepted_step_size is None else accepted_step_size
        if accepted_step_size < 0.0:
            raise ValueError("accepted_step_size must be non-negative")
        maximum_gap = expected_cadence + accepted_step_size + 1.0e-10
        if np.any(np.diff(span_times) > maximum_gap):
            raise ValueError("native VLM clock has a gap beyond its declared cadence")
    recorded_end = float(span_times[-1])
    if end_time is None:
        end_time = recorded_end
    else:
        end_time = float(end_time)
        if not np.isfinite(end_time) or recorded_end < end_time - 1.0e-10:
            raise ValueError("native VLM histories do not cover the recorded native horizon")
        if recorded_end > end_time + 1.0e-10:
            raise ValueError("native VLM histories extend beyond the recorded native horizon")
    cutoff = end_time - revolutions * rotation_period
    if span_times[0] > cutoff + max(1.0e-10, expected_cadence or 0.0):
        raise ValueError("native VLM histories do not cover the requested final window")
    span_window = span[span.time >= cutoff - 1.0e-12].copy()
    chord_window = chord[chord.time >= cutoff - 1.0e-12].copy()
    window_span_steps, window_span_times = _native_vlm_clock(span_window, "spanwise window")
    window_chord_steps, window_chord_times = _native_vlm_clock(chord_window, "chordwise window")
    if not np.array_equal(window_span_steps, window_chord_steps) or not np.allclose(
        window_span_times, window_chord_times, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("spanwise and chordwise averaging windows do not match")
    if window_span_times[-1] < end_time - 1.0e-10:
        raise ValueError("native VLM averaging window does not reach the recorded horizon")
    return span_window, chord_window, cutoff, end_time


def main():
    args = build_arg_parser(__doc__).parse_args()
    inputs = rotor_inputs()
    vlm = inputs.metadata["configuration"]["numerics"]["vlm"]
    surface = vlm["surfaces"][0]["name"]
    span = pd.read_csv(inputs.samples_dir / f"vlm_spanwise_{surface}.csv")
    chord = pd.read_csv(inputs.samples_dir / f"vlm_chordwise_{surface}.csv")
    expected_cadence = _native_vlm_logging_cadence(
        inputs.metadata,
        inputs.time_step_size,
    )
    expected_chord_panels = _native_vlm_expected_chord_panels(vlm["surfaces"][0])
    state_time = float(inputs.metadata["state"]["time"])
    native_end_time = float(span["time"].max())
    if (
        state_time < native_end_time - 1.0e-10
        or state_time > native_end_time + expected_cadence + 1.0e-10
    ):
        raise ValueError(
            "native VLM history horizon is inconsistent with the recorded solver state"
        )
    span, chord, cutoff, end_time = shared_vlm_window(
        span,
        chord,
        rotation_period=inputs.rotation_period,
        end_time=native_end_time,
        expected_cadence=expected_cadence,
        accepted_step_size=inputs.time_step_size,
        expected_chord_panels=expected_chord_panels,
    )
    start, end = cutoff / inputs.rotation_period, end_time / inputs.rotation_period
    positions = chord.groupby(["step", "station_id"])[["bound_y", "bound_z"]].mean()
    positions["radius"] = np.linalg.norm(positions, axis=1)
    sampled = span.merge(
        positions[["radius"]],
        on=["step", "station_id"],
        how="left",
        validate="one_to_one",
    )
    if sampled["radius"].isna().any():
        raise ValueError("chordwise sectional positions are missing for spanwise keys")
    sampled = (
        sampled.groupby("station_id")
        .agg(
            radius=("radius", "mean"),
            circulation=("circulation_magnitude", "mean"),
            cl=("section_lift_coefficient_from_circulation", "mean"),
        )
        .sort_values("radius")
    )
    bem = bem_reference()
    colors, theme = load_theme()
    fig, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), constrained_layout=True)
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
            color=colors["VPMpurple"],
            label=f"Mean, rev {start:.1f}–{end:.1f}",
        )
        axis.plot(
            bem.normalized_radial_position, reference, "--", color=colors["reference"], label="BEM"
        )
        axis.set_xlabel(r"Radius, $r/R$")
        axis.legend()
    axes[0].set(ylabel=r"Circulation, $\Gamma/(U_\infty R)$", title="Blade circulation")
    axes[1].set(ylabel=r"Section lift, $c_l$", title="Local aerodynamic loading")
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_loading_validation.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )


if __name__ == "__main__":
    main()
