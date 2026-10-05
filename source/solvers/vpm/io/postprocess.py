"""Read accepted native VPM histories and coupled visualization state."""

from pathlib import Path
import warnings

import h5py
import numpy as np

from source.solution_layout import vpm_backup_files


def accepted_history(data, metadata):
    """Return rows through the solver's recorded accepted state."""
    state = metadata["state"]
    horizon = float(state["time"])
    dt = float(metadata["configuration"]["numerics"]["time_step_size"])
    tolerance = max(1e-10, dt * 1e-6)
    if not np.isfinite(horizon) or dt <= 0 or not np.isfinite(dt):
        raise ValueError("invalid recorded accepted horizon")
    times = data.time.to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("native history has non-finite timestamps")
    if "step" in data:
        steps = data.step.to_numpy()
        current = steps >= state["initial_step"]
        expected = state["initial_time"] + (steps - state["initial_step"]) * dt
        if (
            not np.isfinite(steps).all()
            or np.any(steps < 0)
            or np.any(steps != np.rint(steps))
            or not np.allclose(times[current], expected[current], rtol=0, atol=tolerance)
            or np.any(times[~current] > state["initial_time"] + tolerance)
        ):
            raise ValueError("native history step/time clock disagrees with recorded solver clock")
        accepted = (steps <= state["step"]) & (times <= horizon + tolerance)
    else:
        accepted = times <= horizon + tolerance
    if not np.any(accepted):
        raise ValueError("native history has no rows within the accepted horizon")
    if not np.all(accepted):
        warnings.warn(
            f"Plotting accepted history through t={horizon:.9g} s; newer sampler rows are retained on disk and excluded from this plot.",
            stacklevel=2,
        )
    return data.loc[accepted].copy()


def backup_clock(path):
    """Read the accepted clock from one native numerical checkpoint."""
    with h5py.File(path, "r") as archive:
        solver = archive["solver"]
        step, time = int(solver.attrs["step"]), float(solver.attrs["time"])
    if step < 0 or not np.isfinite(time):
        raise ValueError(f"Invalid native checkpoint clock: {path}")
    return step, time


def backup_frames(solution_dir):
    """Return ordered native checkpoints as ``(time, path)`` pairs."""
    paths = vpm_backup_files(Path(solution_dir))
    clocks = [backup_clock(path) for path in paths]
    records = [(time, path) for (_, time), path in zip(clocks, paths, strict=True)]
    times = np.asarray([time for _, time in clocks])
    steps = np.asarray([step for step, _ in clocks])
    if not records or np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError(f"Native checkpoint clocks are missing or unordered: {solution_dir}")
    return records


def vlm_surface(path):
    """Read coupled panel corners and bound circulation from native state."""
    with h5py.File(path, "r") as archive:
        vlm = archive["solver/vlm"]
        corners = np.asarray(vlm["panel_corner_position"], dtype=float)
        circulation = np.asarray(vlm["circulation"], dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 3):
        raise ValueError(f"Invalid native VLM panel corners: {path}")
    if circulation.shape != (len(corners),):
        raise ValueError(f"Inconsistent native VLM panel arrays: {path}")
    if not np.isfinite(corners).all() or not np.isfinite(circulation).all():
        raise ValueError(f"Non-finite native VLM state: {path}")
    return corners, circulation


def loading_clock(data, label):
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


def loading_station_keys(
    data,
    label,
    *,
    panel_column=None,
    expected_panel_indices=None,
):
    """Validate native station keys, retaining chordwise panel identity.

    Spanwise histories have one row per ``(step, station_id)``.  Chordwise
    histories intentionally have several rows for that same station, so their
    unique row key is ``(step, station_id, chord_index)``.  The grouped
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


def chord_panel_indices(surface):
    """Read a uniform chord-panel conditions from the native VLM surface mesh."""
    segments = [segment for wing in surface["geometry"]["wings"] for segment in wing["segments"]]
    counts = {int(segment["n_chordwise_panels"]) for segment in segments}
    if len(counts) != 1:
        return None
    count = counts.pop()
    if count < 1:
        raise ValueError("native VLM surface has no chordwise panels")
    return tuple(range(count))


def loading_window(
    span,
    chord,
    *,
    duration,
    end_time=None,
    expected_cadence=None,
    accepted_step_size=None,
    expected_chord_panels=None,
    cadence_start_time=None,
):
    """Select one shared final window after matching span/chord clocks."""
    if duration <= 0:
        raise ValueError("window duration must be positive")
    span_steps, span_times = loading_clock(span, "spanwise")
    chord_steps, chord_times = loading_clock(chord, "chordwise")
    if not np.array_equal(span_steps, chord_steps) or not np.allclose(
        span_times, chord_times, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("spanwise and chordwise VLM clocks do not match")
    span_keys = loading_station_keys(span, "spanwise")
    chord_keys = loading_station_keys(
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
        gaps = np.diff(span_times)
        if cadence_start_time is not None:
            # A continuation can change dt: its recorded schedule constrains
            # this segment, while older native clocks remain paired and ordered.
            gaps = gaps[span_times[:-1] >= cadence_start_time - 1.0e-10]
        if np.any(gaps > maximum_gap):
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
    cutoff = end_time - duration
    if span_times[0] > cutoff:
        raise ValueError("native VLM histories do not cover the requested final window")
    # Retain the lower bracketing snapshot for an exact physical-time mean.
    first = span_times[np.searchsorted(span_times, cutoff, side="right") - 1]
    span_window = span[span.time >= first].copy()
    chord_window = chord[chord.time >= first].copy()
    window_span_steps, window_span_times = loading_clock(span_window, "spanwise window")
    window_chord_steps, window_chord_times = loading_clock(chord_window, "chordwise window")
    if not np.array_equal(window_span_steps, window_chord_steps) or not np.allclose(
        window_span_times, window_chord_times, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("spanwise and chordwise averaging windows do not match")
    if window_span_times[-1] < end_time - 1.0e-10:
        raise ValueError("native VLM averaging window does not reach the recorded horizon")
    return span_window, chord_window, cutoff, end_time


def particle_state(path):
    """Read active native particle fields, solver clock and attached VLM arrays."""
    with h5py.File(path, "r") as archive:
        solver = archive["solver"]
        count = int(solver.attrs["n_particles_total"])
        data = {name: np.asarray(array[:count]) for name, array in archive["particles"].items()}
        data["step"] = int(solver.attrs["step"])
        data["time"] = float(solver.attrs["time"])
        if "vlm" in solver:
            data["vlm"] = {
                name: np.asarray(array)
                for name, array in solver["vlm"].items()
                if isinstance(array, h5py.Dataset)
            }
    if count < 0 or data["step"] < 0 or not np.isfinite(data["time"]):
        raise ValueError(f"Invalid native particle clock or count: {path}")
    for name, shape in (
        ("position", (count, 3)),
        ("vortex_strength", (count, 3)),
        ("core_radius", (count,)),
    ):
        values = data[name]
        if values.shape != shape or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"Invalid native particle {name}: {path}")
    if np.any(data["core_radius"] <= 0):
        raise ValueError(f"Invalid native particle core_radius: {path}")
    return data


def coupled_frames(solution_dir):
    """Read accepted native backups with attached surface geometry."""
    records = backup_frames(solution_dir)
    metadata_path = Path(solution_dir) / "vpm_metadata.json"
    if metadata_path.is_file():
        import json

        state = json.loads(metadata_path.read_text())["state"]
        records = [
            (time, path)
            for time, path in records
            if time <= state["time"] + 1e-10 and backup_clock(path)[0] <= state["step"]
        ]
    if not records:
        raise ValueError("no coupled backups within the recorded accepted horizon")
    for _, path in records:
        vlm_surface(path)
    return records


def surface_series(path):
    """Read a native surface history with one finite, ordered clock and grid."""
    from defusedxml import ElementTree
    import pyvista as pv

    path = Path(path)
    frames = ElementTree.parse(path).findall(".//DataSet")
    times = np.asarray([float(frame.attrib["timestep"]) for frame in frames])
    if not len(times) or not np.isfinite(times).all():
        raise ValueError(f"{path}: empty or non-finite native surface clock")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: unordered native surface clock")
    grids = [pv.read(path.parent / frame.attrib["file"]) for frame in frames]
    points = np.asarray(grids[0].points)
    if not np.isfinite(points).all() or any(
        not np.array_equal(grid.points, points) for grid in grids
    ):
        raise ValueError(f"{path}: native sampling grid is non-finite or changes")
    velocity = np.asarray([grid["velocity"] for grid in grids])
    if velocity.shape != (len(times), len(points), 3) or not np.isfinite(velocity).all():
        raise ValueError(f"{path}: non-finite or inconsistent native velocity fields")
    return times, points, velocity


def impulse_history(
    force,
    integrals,
    *,
    density,
    time_step_size,
    flow_interval_steps=None,
    flow_interval_time=None,
    start_time,
    end_time,
):
    """Compare native surface loads and coupled impulse on identical accepted clocks.

    Forces are fluid-on-body. Fluid impulse and the recorded relaxation transfer
    are per density. Keep the raw balance and numerical transfer separate: removing
    a stabilization source from an accounting residual does not validate that source.
    """
    force_fields = [f"{prefix}_{axis}" for prefix in ("force", "unsteady_force") for axis in "xyz"]
    flow_fields = [
        f"{prefix}_{axis}"
        for prefix in ("coupled_linear_impulse", "pedrizzetti_cumulative_linear_impulse_transfer")
        for axis in "xyz"
    ]
    for name, data, columns in (("force", force, force_fields), ("flow", integrals, flow_fields)):
        missing = {"time", *columns} - set(data.columns)
        if missing:
            raise ValueError(f"Native {name} history lacks required columns: {sorted(missing)}")
        if len(data) < 2 or not np.isfinite(data[["time", *columns]].to_numpy()).all():
            raise ValueError(f"Native {name} history is incomplete or non-finite")
        if np.any(np.diff(data.time) <= 0):
            raise ValueError(f"Native {name} clocks must increase strictly")
    clock = force.time.to_numpy()
    if not np.allclose(np.diff(clock), time_step_size, rtol=1e-7, atol=1e-10):
        raise ValueError("Impulse validation requires surface loads at every accepted time step")
    if (flow_interval_steps is None) == (flow_interval_time is None):
        raise ValueError("Specify exactly one native flow cadence: steps or physical time")
    flow_cadence = (
        flow_interval_steps * time_step_size
        if flow_interval_steps is not None
        else float(flow_interval_time)
    )
    if not np.isfinite(flow_cadence) or flow_cadence <= 0.0:
        raise ValueError("Native flow cadence must be finite and positive")
    if not np.allclose(np.diff(integrals.time), flow_cadence, rtol=1e-7, atol=1e-10):
        raise ValueError("Native flow samples have gaps in their configured cadence")
    if (
        integrals.time.iloc[0] > start_time + flow_cadence
        or integrals.time.iloc[-1] < end_time - flow_cadence - 1e-10
    ):
        raise ValueError("Native flow samples do not cover the requested impulse window")
    selected = integrals[(integrals.time >= start_time) & (integrals.time <= end_time)]
    times = selected.time.to_numpy()
    if len(times) < 2 or times[0] < clock[0] or times[-1] > clock[-1]:
        raise ValueError("Force and flow histories do not cover the requested impulse window")
    # Search both neighbours to tolerate CSV roundoff without interpolating an
    # unsteady pressure difference onto a different physical interval.
    indices = np.searchsorted(clock, times).clip(0, len(clock) - 1)
    left = np.maximum(indices - 1, 0)
    indices = np.where(abs(clock[left] - times) < abs(clock[indices] - times), left, indices)
    if not np.allclose(clock[indices], times, rtol=0, atol=1e-8 * time_step_size):
        raise ValueError("Flow samples do not coincide with accepted force clocks")
    total = force[[f"force_{a}" for a in "xyz"]].to_numpy()
    pressure = force[[f"unsteady_force_{a}" for a in "xyz"]].to_numpy()
    kj = total - pressure
    intervals = np.diff(clock)[:, None] * (0.5 * (kj[:-1] + kj[1:]) + pressure[1:])
    cumulative = np.vstack((np.zeros(3), np.cumsum(intervals, axis=0)))
    loads = cumulative[indices] - cumulative[indices[0]]
    fluid = -density * selected[[f"coupled_linear_impulse_{a}" for a in "xyz"]].to_numpy()
    relaxation = (
        -density
        * selected[
            [f"pedrizzetti_cumulative_linear_impulse_transfer_{a}" for a in "xyz"]
        ].to_numpy()
    )
    return times, loads, fluid - fluid[0], relaxation - relaxation[0]


def profile_mean(data, start, end, *, coordinates, fields):
    """Mean one fixed native point table over a fully bracketed time window."""
    from openonda.validation import time_mean

    required = {"time", *coordinates, *fields}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Native profile lacks fields: {sorted(missing)}")
    times = np.sort(data.time.unique())
    frames = [data[data.time == time].sort_values(list(coordinates)) for time in times]
    if not frames or any(frame.duplicated(list(coordinates)).any() for frame in frames):
        raise ValueError("native profile has missing or duplicate point positions")
    positions = frames[0][list(coordinates)].to_numpy(float)
    if not np.isfinite(positions).all() or any(
        not np.array_equal(frame[list(coordinates)].to_numpy(float), positions)
        for frame in frames[1:]
    ):
        raise ValueError("native profile has changing or non-finite positions")
    values = np.stack([frame[list(fields)].to_numpy(float) for frame in frames])
    return positions, time_mean(times, values, start, end)


def surface_mean(frames, start, end, *, field):
    """Mean native surface frames on their unchanged grid and bracketed clock."""
    import pyvista as pv

    from openonda.validation import time_mean

    times = np.asarray([time for time, _ in frames], dtype=float)
    grids = [pv.read(path) for _, path in frames]
    if not grids:
        raise ValueError("native surface has no frames")
    points = np.asarray(grids[0].points)
    if not np.isfinite(points).all() or any(
        not np.array_equal(grid.points, points) for grid in grids[1:]
    ):
        raise ValueError("native surface has changing or non-finite grid points")
    fields = np.asarray([grid[field] for grid in grids])
    return points, time_mean(times, fields, start, end)


def common_frame_time(*series):
    """Select the latest physical clock represented by every native series."""
    from openonda.saved_times import match_saved_times

    matched = match_saved_times(*series)
    if not matched.times:
        raise ValueError("Native series have no common saved physical state")
    return matched.times[-1]


def saved_frame(frames, time=None):
    """Select an existing native state at the supplied physical clock."""
    from openonda.saved_times import match_saved_times

    frames = sorted(frames, key=lambda frame: frame[0])
    clocks = [time for time, _ in frames]
    requested = [time] if time is not None else clocks[-1:]
    match = match_saved_times(requested, clocks)
    if not match.times:
        raise ValueError("Requested physical state is absent from native frames")
    return frames[match.indices[1][0]]
