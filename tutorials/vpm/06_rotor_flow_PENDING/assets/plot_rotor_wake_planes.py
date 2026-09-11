#!/usr/bin/env python3
"""Compare native downstream velocity samples with an ideal actuator-disk wake."""

from __future__ import annotations

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from defusedxml import ElementTree
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from openonda import plotting as theme
from openonda.rotor_theory import (
    actuator_disk_velocity_ratio,
    axial_induction_factor_from_thrust_coefficient,
)
from ._common import (
    FIGURES_DIR,
    FIELD_STATIONARITY_COMPARISON_REVOLUTIONS,
    OPERATING_WINDOW_REVOLUTIONS,
    SIGNAL_ONSET_PERSISTENCE_FRAMES,
    SIGNAL_ONSET_RELATIVE_THRESHOLD,
    bem_reference,
    build_arg_parser,
    read_operating_point,
    rotor_inputs,
)
from .finite_distance_theory import build_system, induced_velocity


def relative_drift(values):
    """Compare two halves of a time window, normalized by its mean magnitude."""
    values = np.asarray(values)
    half = len(values) // 2
    return abs(values[:half].mean() - values[half:].mean()) / max(abs(values.mean()), 1e-12)


def _interpolate_native_value(times, values, target):
    """Interpolate a recorded value at a bracketed boundary, never extrapolate."""
    tolerance = max(1.0e-12, abs(target) * 1.0e-12)
    index = int(np.searchsorted(times, target))
    if index < len(times) and abs(times[index] - target) <= tolerance:
        return values[index]
    if index == 0 or index == len(times):
        raise ValueError(f"native history does not bracket boundary t={target:.17g}")
    lower, upper = index - 1, index
    fraction = (target - times[lower]) / (times[upper] - times[lower])
    return values[lower] + fraction * (values[upper] - values[lower])


def _time_weighted_mean(times, values, start, stop):
    """Integrate native values over an exactly bracketed physical-time interval."""
    if stop <= start:
        raise ValueError("native time-weighted interval must be positive")
    start_value = _interpolate_native_value(times, values, start)
    stop_value = _interpolate_native_value(times, values, stop)
    inside = (times > start) & (times < stop)
    clock = np.r_[start, times[inside], stop]
    sampled_values = np.concatenate(
        (start_value[None, ...], values[inside], stop_value[None, ...]), axis=0
    )
    return np.trapezoid(sampled_values, clock, axis=0) / (stop - start)


def induced_field_drift(
    times,
    velocity,
    background,
    period,
    comparison_rotations=FIELD_STATIONARITY_COMPARISON_REVOLUTIONS,
    *,
    window_start=None,
    window_end=None,
):
    """Compare first-two versus final-three whole-revolution means.

    For the five-revolution native window, the comparison covers all included
    time: two complete revolutions at the front versus three at the end. When
    explicit boundaries are supplied, every boundary must be bracketed by
    recorded native frames; no extrapolation is allowed.
    """
    comparison_rotations = int(comparison_rotations)
    if comparison_rotations < 2:
        raise ValueError("comparison_rotations must be at least two")
    if window_start is None:
        window_end = float(times[-1]) if window_end is None else float(window_end)
        window_start = window_end - comparison_rotations * period
    elif window_end is None:
        raise ValueError("window_start and window_end must be supplied together")
    else:
        window_start, window_end = float(window_start), float(window_end)
    expected_span = comparison_rotations * period
    if window_end - window_start < expected_span - max(1.0e-10, expected_span * 1.0e-10):
        raise ValueError("native history does not cover the requested whole-revolution window")
    early_rotations = comparison_rotations // 2
    split = window_start + early_rotations * period
    early = _time_weighted_mean(times, velocity, window_start, split)
    late = _time_weighted_mean(times, velocity, split, window_end)
    scale = np.linalg.norm(0.5 * (early + late) - background)
    if scale <= 1e-12:
        return np.nan, comparison_rotations
    return np.linalg.norm(late - early) / scale, comparison_rotations


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
    files = sorted(solution_dir.glob("vpm_*.h5"))
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
        for left, right in zip(checkpoints, checkpoints[1:])
    ):
        return status_records(
            "invalid_order", reason="checkpoint time/step order is not increasing"
        )
    records = []
    for station_x in station_positions:
        station_x = float(station_x)
        bracket = None
        for previous, current in zip(checkpoints, checkpoints[1:]):
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
    window_start = p.metadata["state"]["time"] - OPERATING_WINDOW_REVOLUTIONS * p.rotation_period
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
    end, start = state["time"], state["time"] - rotations * p.rotation_period
    declared = [item for item in config["samplers"]["items"] if item["type"] == "SurfaceSampler"]
    if required_names is not None:
        required_names = tuple(required_names)
        declared_by_name = {item["file_name"]: item for item in declared}
        missing = [name for name in required_names if name not in declared_by_name]
        if missing:
            raise ValueError(f"Missing required downstream planes: {missing}")
        declared = [declared_by_name[name] for name in required_names]
    if not declared:
        raise ValueError("No downstream planes declared in native metadata")
    steps = np.arange(state["initial_step"] + 1, state["step"] + 1)
    accepted_times = state["initial_time"] + (steps - state["initial_step"]) * dt
    background = np.asarray(config["numerics"]["freestream_velocity"])
    records = []
    for item in declared:
        name = item["file_name"]
        pvd = p.samples_dir / f"{name}.pvd"
        frames = ElementTree.parse(pvd).findall(".//DataSet")
        times = np.array([float(frame.attrib["timestep"]) for frame in frames])
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name}: missing, duplicate or unordered frame times")
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
        complete = (
            len(expected) >= 4
            and len(selected) == len(expected)
            and np.allclose(sampled_times, expected, rtol=0, atol=tolerance)
            and expected[0] <= start + cadence + tolerance
            and end - expected[-1] <= cadence + tolerance
            and times[-1] <= end + tolerance
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
            dict(
                name=name,
                times=sampled_times,
                points=points,
                velocity=velocity,
                complete=complete,
                induced_field_drift=drift,
                compared_rotations=compared_rotations,
                window_mean_velocity=window_mean_velocity,
            )
        )
    if require_complete:
        station_radius = getattr(p, "station_radius", p.rotor_radius)
        stations = {row["name"]: row["points"][0, 0] / (2 * station_radius) for row in records}
        if required_names is None:
            expected_stations = {1.0, 2.0}
            if (
                len(stations) != 2
                or set(np.round(np.asarray(tuple(stations.values())), 6)) != expected_stations
            ):
                raise ValueError("Rotor wake series has unexpected declared 1D/2D stations")
        else:
            expected = {"wake_1D": 1.0, "wake_2D": 2.0}
            if set(stations) != set(required_names) or any(
                not np.isclose(stations[name], expected.get(name, np.nan), rtol=0, atol=1e-6)
                for name in required_names
            ):
                raise ValueError("Rotor validation requires native downstream planes at 1D and 2D")
    return records


def plane_profiles(p, rotations=OPERATING_WINDOW_REVOLUTIONS):
    """Time/azimuth averages of the solver's published native velocity planes."""
    records = []
    for plane in native_plane_windows(p, rotations):
        points = plane["points"]
        # Use only complete annuli within the native square sampling plane.
        extent = min(-points[:, 1:].min(axis=0).max(), points[:, 1:].max(axis=0).min())
        edges = np.linspace(0, extent / p.rotor_radius, 49)
        radius = 0.5 * (edges[1:] + edges[:-1])
        r = np.linalg.norm(points[:, 1:], axis=1) / p.rotor_radius
        velocity = plane["velocity"][:, :, 0] / p.freestream_speed
        indices = np.digitize(r, edges) - 1
        valid = (indices >= 0) & (indices < len(radius))
        counts = np.bincount(indices[valid], minlength=len(radius))
        if "window_mean_velocity" in plane:
            mean_velocity = plane["window_mean_velocity"][:, 0] / p.freestream_speed
        else:
            mean_velocity = np.trapezoid(velocity, plane["times"], axis=0) / np.ptp(plane["times"])
        sums = np.bincount(indices[valid], weights=mean_velocity[valid], minlength=len(radius))
        records.append(
            dict(
                name=plane["name"],
                x=points[0, 0],
                radius=radius,
                mean=np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0),
                times=plane["times"],
                complete=plane["complete"],
                induced_field_drift=plane["induced_field_drift"],
                compared_rotations=plane["compared_rotations"],
                **_signal_onset_window_status(p, plane),
            )
        )
    return sorted(records, key=lambda row: row["x"])


def finite_distance_profiles(p, rotations=OPERATING_WINDOW_REVOLUTIONS):
    """Return native axial/azimuthal profiles beside the right-cylinder theory.

    The native velocity is time-averaged at each fixed plane point before
    annular binning.  The reference uses the matched BEM circulation table and
    the analytical finite-distance cylinder functions; it is deliberately not
    fitted to the native samples.
    """
    bem = bem_reference()
    system = build_system(
        bem,
        number_of_blades=p.n_blades,
        freestream_speed=p.freestream_speed,
        angular_velocity=p.angular_velocity,
        hub_radius=p.hub_radius,
        rotor_radius=p.rotor_radius,
    )
    configuration = p.metadata["configuration"]
    signed_angular_velocity = float(
        configuration["numerics"]["vlm"]["surfaces"][0]["kinematics"]["angular_speed"]
    )
    rotation_sign = np.sign(signed_angular_velocity) or 1.0
    records = []
    for plane in native_plane_windows(p, rotations):
        points = plane["points"]
        extent = min(-points[:, 1:].min(axis=0).max(), points[:, 1:].max(axis=0).min())
        edges = np.linspace(0.0, extent / p.rotor_radius, 49)
        radius = 0.5 * (edges[1:] + edges[:-1])
        point_radius = np.linalg.norm(points[:, 1:], axis=1)
        radial_coordinate = point_radius / p.rotor_radius
        indices = np.digitize(radial_coordinate, edges) - 1
        valid = (indices >= 0) & (indices < len(radius)) & (point_radius > 1.0e-10)
        counts = np.bincount(indices[valid], minlength=len(radius))
        velocity = plane["velocity"]
        mean_velocity = plane.get("window_mean_velocity")
        if mean_velocity is None:
            mean_velocity = np.trapezoid(velocity, plane["times"], axis=0) / np.ptp(plane["times"])
        axial = mean_velocity[:, 0]
        tangential = -points[:, 2] * mean_velocity[:, 1] + points[:, 1] * mean_velocity[:, 2]
        tangential = np.divide(
            tangential,
            point_radius,
            out=np.full_like(tangential, np.nan, dtype=float),
            where=point_radius > 1.0e-10,
        )

        def bin_mean(values):
            sums = np.bincount(
                indices[valid], weights=np.asarray(values)[valid], minlength=len(radius)
            )
            return np.divide(
                sums,
                counts,
                out=np.full(len(radius), np.nan, dtype=float),
                where=counts > 0,
            )

        actual_axial_velocity = bin_mean(axial)
        actual_axial = 1.0 - actual_axial_velocity / p.freestream_speed
        actual_tangential = -bin_mean(tangential) / (
            signed_angular_velocity * radius * p.rotor_radius
        )
        reference_axial = np.full_like(radius, np.nan, dtype=float)
        reference_axial_velocity = np.full_like(radius, np.nan, dtype=float)
        reference_tangential = np.full_like(radius, np.nan, dtype=float)
        for index, normalized_radius in enumerate(radius):
            if not np.isfinite(actual_axial[index]):
                continue
            prediction = induced_velocity(
                normalized_radius * p.rotor_radius,
                points[0, 0],
                system,
                freestream_speed=p.freestream_speed,
                angular_velocity=p.angular_velocity,
            )
            reference_axial_velocity[index] = p.freestream_speed + prediction["axial_velocity"]
            reference_axial[index] = prediction["axial_induction"]
            reference_tangential[index] = prediction["tangential_induction"]
        reference_tangential_velocity = (
            -reference_tangential * p.angular_velocity * radius * p.rotor_radius * rotation_sign
        )
        actual_tangential_velocity = bin_mean(tangential)
        # Retain signed velocities as well as induction factors so the plot can
        # state the coordinate convention without hiding a sign error.
        records.append(
            {
                "name": plane["name"],
                "x": points[0, 0],
                "radius": radius,
                "actual_axial_induction": actual_axial,
                "reference_axial_induction": reference_axial,
                "actual_axial_velocity": actual_axial_velocity,
                "reference_axial_velocity": reference_axial_velocity,
                "actual_tangential_induction": actual_tangential,
                "reference_tangential_induction": reference_tangential,
                "actual_tangential_velocity": actual_tangential_velocity,
                "reference_tangential_velocity": reference_tangential_velocity,
                "complete": plane["complete"],
                "induced_field_drift": plane["induced_field_drift"],
                "compared_rotations": plane["compared_rotations"],
                **_signal_onset_window_status(p, plane),
            }
        )
    return sorted(records, key=lambda row: row["x"])


def main():
    args = build_arg_parser(__doc__).parse_args()
    theme.set_thesis_style()
    p = rotor_inputs()
    ct, _ = read_operating_point()
    induction = axial_induction_factor_from_thrust_coefficient(ct)
    fig, ax = plt.subplots(figsize=theme.figure_size("stacked"), constrained_layout=True)
    for row in plane_profiles(p):
        drift = row["induced_field_drift"]
        start, end = row["times"][[0, -1]] / p.rotation_period
        quality = (
            f"field drift {100 * drift:.1f}\\%" if np.isfinite(drift) else "wake drift unqualified"
        )
        if not row["signal_onset_qualifies"]:
            quality += "; signal onset unqualified"
        station_radius = getattr(p, "station_radius", p.rotor_radius)
        label = f"{row['x'] / (2 * station_radius):g}D; rev {start:.1f}–{end:.1f}\n{quality}"
        if not row["complete"]:
            label += "; incomplete"
        ax.plot(
            row["radius"],
            row["mean"],
            ls="-" if row["complete"] and row["signal_onset_qualifies"] and drift <= 0.01 else ":",
            label=label,
        )
        status = f"{drift:.2%}" if np.isfinite(drift) else "unqualified"
        print(
            f"{row['name']}: {len(row['times'])} frames; induced-field drift {status} "
            f"over {row['compared_rotations']} revolutions; "
            f"signal onset={'qualified' if row['signal_onset_qualifies'] else 'unqualified'}"
        )
    ax.axhline(1, color="0.5", lw=0.7, label="Freestream")
    if np.isfinite(induction) and induction < 0.5:
        r = np.linspace(0, 2.0, 500)
        ax.plot(
            r,
            actuator_disk_velocity_ratio(induction, r),
            color="0.2",
            ls="--",
            label=f"Ideal far wake, CT={ct:.3f}",
        )
        ax.plot([0, 1], [1 - induction] * 2, color="0.5", ls=":", label="Ideal disk velocity")
    else:
        ax.text(
            0.02, 0.03, f"CT={ct:.3f}: outside ideal low-induction branch", transform=ax.transAxes
        )
    ax.set(
        xlabel=r"$r/R$",
        ylabel=r"$\langle u_x/U_\infty\rangle_{t,\theta}$",
        xlim=(0, 2.0),
        title="Time and azimuthal mean",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_wake_planes.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
