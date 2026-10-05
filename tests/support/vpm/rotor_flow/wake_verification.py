"""Native sampling coverage, onset and checkpoint evidence for rotor verification."""

from defusedxml import ElementTree
import h5py
import numpy as np
import pyvista as pv

from source.solution_layout import vpm_backup_files
from tests._tutorial_helpers import load_tutorial_module

_physics = load_tutorial_module("vpm/rotor_flow", "assets.plot_rotor_wake_planes")
_common = load_tutorial_module("vpm/rotor_flow", "assets._common")
induced_field_drift = _physics.induced_field_drift
_time_weighted_mean = _physics._time_weighted_mean
OPERATING_WINDOW_REVOLUTIONS = _common.OPERATING_WINDOW_REVOLUTIONS
SIGNAL_ONSET_RELATIVE_THRESHOLD = 0.01
SIGNAL_ONSET_PERSISTENCE_FRAMES = 3


def assess_wake_signal_onset(
    p,
    name,
    *,
    relative_threshold=SIGNAL_ONSET_RELATIVE_THRESHOLD,
    persistence_frames=SIGNAL_ONSET_PERSISTENCE_FRAMES,
):
    """Assess whether a native plane has persistent induced-velocity signal onset.

    Signal onset is a diagnostic prerequisite, not a convergence pass: the first
    run of ``persistence_frames`` native frames whose RMS induced-vector
    magnitude reaches the fixed fraction of freestream is reported. A run may
    still be unqualified when its final five-revolution window begins before
    this onset time.
    """
    config = p.metadata["configuration"]
    declared = {
        item["file_name"]: item
        for item in config["samplers"]["items"]
        if item["type"] == "SurfaceSampler"
    }
    if name not in declared:
        raise ValueError(f"Missing native downstream plane: {name}")
    pvd = p.samples_dir / f"{name}.pvd"
    frames = ElementTree.parse(pvd).findall(".//DataSet")
    times = np.array([float(frame.attrib["timestep"]) for frame in frames])
    if (
        len(times) < persistence_frames
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(f"{name}: missing, duplicate or unordered frame times")
    grids = [pv.read(pvd.parent / frame.attrib["file"]) for frame in frames]
    points = np.asarray(grids[0].points)
    if not np.isfinite(points).all() or any(
        not np.array_equal(grid.points, points) for grid in grids
    ):
        raise ValueError(f"{name}: velocity-plane geometry changes in native history")
    velocity = np.asarray([np.asarray(grid["velocity"]) for grid in grids])
    if velocity.ndim != 3 or velocity.shape[2] != 3 or not np.isfinite(velocity).all():
        raise ValueError(f"{name}: missing or non-finite velocity vectors")
    background = np.asarray(config["numerics"]["freestream_velocity"], dtype=float)
    background_scale = float(np.linalg.norm(background))
    if not np.isfinite(background_scale) or background_scale <= 0.0:
        raise ValueError(f"{name}: freestream scale is not finite and positive")
    induced_rms = np.sqrt(np.mean(np.sum((velocity - background) ** 2, axis=2), axis=1))
    threshold = float(relative_threshold) * background_scale
    above = induced_rms >= threshold
    onset_index = None
    for index in range(0, len(above) - persistence_frames + 1):
        if np.all(above[index : index + persistence_frames]):
            onset_index = index
            break
    onset_time = None if onset_index is None else float(times[onset_index])
    return {
        "name": name,
        "signal_onset_time": onset_time,
        "signal_onset_frame": onset_index,
        "threshold": threshold,
        "relative_threshold": float(relative_threshold),
        "persistence_frames": int(persistence_frames),
        "times": times,
        "induced_rms": induced_rms,
    }


def checkpoint_particle_front_brackets(p, station_positions=None):
    """Bracket particle-cloud crossings of nominal stations from native HDF5 backups.

    This is conservative particle-front evidence only. It does not establish
    that the velocity signal at a plane is caused by a convected wake front,
    so a bracket is reported diagnostically and never upgrades signal onset to
    physical arrival.
    """
    solution_dir = getattr(p, "solution_dir", None)
    if station_positions is None:
        station_radius = getattr(p, "station_radius", None)
        if station_radius is None:
            station_radius = p.rotor_radius
        station_positions = (2.0 * station_radius, 4.0 * station_radius)
    station_positions = tuple(float(station_x) for station_x in station_positions)

    def status_records(status, **details):
        return [
            {"station_x": float(station_x), "status": status, **details}
            for station_x in station_positions
        ]

    if solution_dir is None:
        return status_records("unavailable")
    files = vpm_backup_files(solution_dir)
    if len(files) < 2:
        return status_records("unavailable")
    checkpoints = []
    try:
        for path in files:
            with h5py.File(path, "r") as archive:
                solver = archive.get("solver")
                positions = archive.get("particles/position")
                if solver is None or positions is None:
                    raise ValueError(f"{path}: missing solver or particle positions")
                time = float(solver.attrs["time"])
                step = int(solver.attrs["step"])
                if not np.isfinite(time):
                    raise ValueError(f"{path}: checkpoint time is non-finite")
                # Read only streamwise coordinates; this diagnostic does not need
                # the full particle cloud or velocity fields.
                shape = positions.shape
                if len(shape) != 2 or shape[1] < 1:
                    raise ValueError(f"{path}: particle positions have invalid shape")
                values = np.asarray(positions[:, 0], dtype=float)
                if len(values) == 0:
                    maximum = -np.inf
                elif not np.isfinite(values).all():
                    raise ValueError(f"{path}: particle positions are non-finite")
                else:
                    maximum = float(values.max())
                checkpoints.append((time, step, maximum, path))
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        return status_records("invalid", reason=str(exc))

    # File order is the native checkpoint order. Sorting by time would turn a
    # malformed or shuffled history into apparently valid evidence.
    if any(
        right[0] <= left[0] or right[1] <= left[1]
        for left, right in zip(checkpoints, checkpoints[1:], strict=False)
    ):
        return status_records(
            "invalid_order", reason="checkpoint time/step order is not increasing"
        )
    records = []
    for station_x in station_positions:
        station_x = float(station_x)
        bracket = None
        for previous, current in zip(checkpoints, checkpoints[1:], strict=False):
            if previous[2] < station_x <= current[2]:
                bracket = previous, current
                break
        if bracket is None:
            status = "not_bracketed" if checkpoints[0][2] >= station_x else "not_observed"
            record = {"station_x": station_x, "status": status}
        else:
            previous, current = bracket
            record = {
                "station_x": station_x,
                "status": "bracketed",
                "previous_time": previous[0],
                "crossing_time": current[0],
                "previous_step": previous[1],
                "crossing_step": current[1],
                "previous_max_x": previous[2],
                "crossing_max_x": current[2],
            }
        records.append(record)
    return records


def _signal_onset_window_status(p, plane):
    """Return visual qualification metadata without weakening synthetic tests."""
    if not hasattr(p, "metadata") or not hasattr(p, "samples_dir"):
        return {"signal_onset_time": None, "signal_onset_qualifies": True}
    assessment = assess_wake_signal_onset(p, plane["name"])
    onset_time = assessment["signal_onset_time"]
    window_start = plane.get(
        "window_start",
        p.metadata["state"]["time"] - OPERATING_WINDOW_REVOLUTIONS * p.rotation_period,
    )
    return {
        "signal_onset_time": onset_time,
        "signal_onset_qualifies": onset_time is not None and onset_time <= window_start + 1.0e-10,
    }


def native_plane_windows(
    p,
    rotations=OPERATING_WINDOW_REVOLUTIONS,
    *,
    require_complete=False,
    required_names=None,
):
    """Read native planes and qualify one exact, bracketed five-revolution window."""
    from source.solvers.vpm.io.sampling import EverySteps, EveryTime

    config, state = p.metadata["configuration"], p.metadata["state"]
    dt = config["numerics"]["time_step_size"]
    horizon = state["time"]
    declared = [item for item in config["samplers"]["items"] if item["type"] == "SurfaceSampler"]
    if required_names is not None:
        required_names = tuple(required_names)
        declared_by_name = {item["file_name"]: item for item in declared}
        missing = [name for name in required_names if name not in declared_by_name]
        if missing:
            raise ValueError(f"Missing required rotor planes: {missing}")
        declared = [declared_by_name[name] for name in required_names]
    if not declared:
        raise ValueError("No rotor planes declared in native metadata")
    tolerance = max(1e-10, dt * 1e-6)
    collections = {}
    common_times = None
    for item in declared:
        name = item["file_name"]
        frames = ElementTree.parse(p.samples_dir / f"{name}.pvd").findall(".//DataSet")
        times = np.array([float(frame.attrib["timestep"]) for frame in frames])
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name}: missing, duplicate or unordered frame times")
        collections[name] = frames, times
        available = times[times <= horizon + tolerance]
        if common_times is None:
            common_times = available
        else:
            common_times = np.array(
                [
                    time
                    for time in common_times
                    if np.any(np.isclose(available, time, rtol=0, atol=tolerance))
                ]
            )
    if not len(common_times):
        raise ValueError("Native planes have no shared sample within the accepted horizon")
    end = float(common_times[-1])
    start = end - rotations * p.rotation_period
    steps = np.arange(state["initial_step"] + 1, state["step"] + 1)
    accepted_times = state["initial_time"] + (steps - state["initial_step"]) * dt
    background = np.asarray(config["numerics"]["freestream_velocity"])
    records = []
    for item in declared:
        name = item["file_name"]
        pvd = p.samples_dir / f"{name}.pvd"
        frames, times = collections[name]
        schedule_data = item["schedule"]
        if schedule_data["type"] == "EverySteps":
            schedule = EverySteps(
                schedule_data["interval"],
                first_step=schedule_data.get("first_step"),
                start_time=schedule_data.get("start_time"),
            )
            cadence = schedule.interval * dt
        elif schedule_data["type"] == "EveryTime":
            schedule = EveryTime(
                schedule_data["interval"], start_time=schedule_data.get("start_time", 0.0)
            )
            cadence = schedule.interval + dt
        else:
            raise ValueError(f"{name}: unsupported native plane schedule {schedule_data['type']}")
        tolerance = max(1e-10, dt * 1e-6)
        expected = np.array(
            [
                time
                for step, time in zip(steps, accepted_times, strict=True)
                if time > start + tolerance and schedule.is_due(int(step), float(time), dt)
            ]
        )
        selected = np.flatnonzero((times > start + tolerance) & (times <= end + tolerance))
        sampled_times = times[selected]
        # A shared field endpoint must not hide missing scheduled output near
        # the accepted clock. Check the full due cadence through that clock.
        published = times[(times > start + tolerance) & (times <= horizon + tolerance)]
        complete = (
            len(expected) >= 4
            and len(published) == len(expected)
            and np.allclose(published, expected, rtol=0, atol=tolerance)
            and expected[0] <= start + cadence + tolerance
            and end - expected[-1] <= cadence + tolerance
            and times[-1] <= horizon + tolerance
        )
        if require_complete and not complete:
            raise ValueError(f"{name}: incomplete final {rotations:g}-revolution velocity window")
        if len(selected) < 2:
            raise ValueError(f"{name}: too few native frames in the requested window")
        grids = [pv.read(pvd.parent / frames[index].attrib["file"]) for index in selected]
        points = np.asarray(grids[0].points)
        if not np.isfinite(points).all() or any(
            not np.array_equal(grid.points, points) for grid in grids
        ):
            raise ValueError(f"{name}: velocity-plane geometry changes within the window")
        if require_complete and "bounds" in item:
            declared_bounds = item["bounds"]
            if isinstance(declared_bounds, dict):
                declared_bounds = declared_bounds.get("values")
            declared_bounds = np.asarray(declared_bounds, dtype=float)
            actual_bounds = np.array(
                [points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max()]
            )
            spacing = float(item.get("spacing", 0.0))
            tolerance = max(1.0e-6, 0.1 * spacing)
            if declared_bounds.shape != (4,) or not np.allclose(
                actual_bounds, declared_bounds, rtol=0.0, atol=tolerance
            ):
                raise ValueError(
                    f"{name}: native frame extent does not match its declared compact bounds"
                )
        velocity = np.asarray([np.asarray(grid["velocity"]) for grid in grids])
        if velocity.shape != (len(selected), len(points), 3) or not np.isfinite(velocity).all():
            raise ValueError(f"{name}: missing or non-finite velocity vectors")
        drift_indices = np.arange(max(0, selected[0] - 1), min(len(frames), selected[-1] + 2))
        drift_grids = [
            pv.read(pvd.parent / frames[index].attrib["file"]) for index in drift_indices
        ]
        drift_points = np.asarray(drift_grids[0].points)
        if not np.array_equal(drift_points, points) or any(
            not np.array_equal(grid.points, points) for grid in drift_grids
        ):
            raise ValueError(f"{name}: native boundary frames change plane geometry")
        drift_times = np.array([float(frames[index].attrib["timestep"]) for index in drift_indices])
        drift_velocity = np.asarray([np.asarray(grid["velocity"]) for grid in drift_grids])
        if not np.isfinite(drift_velocity).all() or drift_velocity.shape[1:] != velocity.shape[1:]:
            raise ValueError(f"{name}: missing or non-finite boundary velocity vectors")
        try:
            drift, compared_rotations = induced_field_drift(
                drift_times,
                drift_velocity,
                background,
                p.rotation_period,
                window_start=start,
                window_end=end,
            )
            window_mean_velocity = _time_weighted_mean(drift_times, drift_velocity, start, end)
        except ValueError:
            # A complete cadence without bracketed physical-time boundaries is
            # visible but cannot qualify the exact averaging window.
            drift, compared_rotations = np.nan, 0
            window_mean_velocity = np.full((len(points), 3), np.nan, dtype=float)
        records.append(
            {
                "name": name,
                "times": sampled_times,
                "points": points,
                "velocity": velocity,
                "complete": complete,
                "induced_field_drift": drift,
                "compared_rotations": compared_rotations,
                "window_mean_velocity": window_mean_velocity,
                "bracket_times": drift_times,
                "bracket_velocity": drift_velocity,
                "window_start": start,
                "window_end": end,
            }
        )
    if require_complete:
        station_radius = getattr(p, "station_radius", p.rotor_radius)
        stations = {row["name"]: row["points"][0, 0] / (2 * station_radius) for row in records}
        expected_stations = {
            item["file_name"]: float(item["point"][0]) / (2 * station_radius) for item in declared
        }
        if set(stations) != set(expected_stations) or any(
            not np.isclose(stations[name], expected_stations[name], rtol=0, atol=1e-6)
            for name in expected_stations
        ):
            raise ValueError("Rotor series has unexpected declared plane stations")
    return records
