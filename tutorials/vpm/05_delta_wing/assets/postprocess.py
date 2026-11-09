"""Forces, measured heave cycles and period-mean wakes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from openonda import plotting as _theme
from assets.output_files import read_csv_table, read_json, read_pvd_frames
from openonda.vpm.io.postprocess import surface_mean

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution/delta_wing"
SAMPLES_DIR = CASE_DIR / "samples/delta_wing"
FIGURES_DIR = CASE_DIR / "figures"


def _directories(value):
    return [Path(value)] if isinstance(value, (str, Path)) else list(map(Path, value))


def force_history(samples_dirs=None):
    paths = _directories(samples_dirs) if samples_dirs is not None else [SAMPLES_DIR]
    return pd.concat(
        [pd.DataFrame(read_csv_table(path / "vlm_surface_forces.csv")) for path in paths],
        ignore_index=True,
    ).sort_values(["time", "surface"])


def flow_integrals(samples_dirs=None):
    paths = _directories(samples_dirs) if samples_dirs is not None else [SAMPLES_DIR]
    return pd.concat(
        [pd.DataFrame(read_csv_table(path / "flow_integrals.csv")) for path in paths],
        ignore_index=True,
    ).sort_values("time")


def lift_and_thrust(data, solution_dir=SOLUTION_DIR):
    """Add each wing's lift and thrust coefficients on its own reference area."""
    vlm = read_json(Path(solution_dir) / "vpm_metadata.json")["configuration"]["numerics"]["vlm"]
    velocity = _read_freestream_velocity([Path(solution_dir)])
    speed = np.linalg.norm(velocity)
    area = {surface["name"]: surface["geometry"]["refs"]["area"] for surface in vlm["surfaces"]}
    reference = 0.5 * vlm["density"] * speed**2 * data.surface.map(area)
    thrust = -data[["force_x", "force_y", "force_z"]].to_numpy() @ (velocity / speed)
    return data.assign(
        lift_coefficient=data.force_z / reference, thrust_coefficient=thrust / reference
    )


def motion_period(data):
    """Measure the prescribed period from the solver's sampled heave velocity."""
    front = data[data.surface == "front_wing"].sort_values("time")
    peaks, _ = find_peaks(front.translation_velocity_z)
    return float(np.median(np.diff(front.time.to_numpy()[peaks])))


def last_cycles(data, period, count=3):
    """Yield complete cycles; a sample exactly on the endpoint belongs to the prior cycle."""
    times = np.sort(data.time.unique())
    cadence = np.median(np.diff(times))
    first = max(0, int(np.ceil((times[0] - cadence - 1e-09) / period)))
    last = int(np.floor((data.time.max() + 1e-09) / period))
    for cycle in range(max(first, last - count), last):
        rows = data[
            (data.time > cycle * period + 1e-09) & (data.time <= (cycle + 1) * period + 1e-09)
        ]
        yield (cycle, (rows.time.to_numpy() - cycle * period) / period, rows)


def _save_figure(fig, path, figure_format):
    return _theme.export_figure(fig, path, figure_format=figure_format)


def _wake_average(collections, end, period):
    """Average one measured heave cycle on each native wake plane."""
    return [
        surface_mean(
            [(time, path) for time, path, _ in frames], end - period, end, field="velocity"
        )
        for frames in collections
    ]


def _wake_frames(samples_dirs, plane_name):
    return sorted(
        (
            (time, path, directory.name)
            for directory in samples_dirs
            for time, path in read_pvd_frames(directory / plane_name)
        )
    )


def _read_freestream_velocity(solution_dirs):
    return np.asarray(
        read_json(solution_dirs[0] / "vpm_metadata.json")["configuration"]["numerics"][
            "freestream_velocity"
        ]
    )


def _plot_wake_field(
    samples_arg, destination: Path, figure_format: str, field_name: str, *, solution_dirs=None
) -> None:
    "Export one thesis wake-plane figure from native sampled field samples."
    _theme.set_thesis_style()
    samples_dirs = _directories(samples_arg) if samples_arg is not None else [SAMPLES_DIR]
    solution_dirs = _directories(solution_dirs) if solution_dirs is not None else [SOLUTION_DIR]
    velocity = _read_freestream_velocity(solution_dirs)
    speed = np.linalg.norm(velocity)
    direction = velocity / speed
    period = motion_period(force_history(samples_dirs))
    plane_names = sorted((path.name for path in samples_dirs[0].glob("wake_*span.pvd")))
    collections = [_wake_frames(samples_dirs, name) for name in plane_names]
    end = min((frames[-1][0] for frames in collections))
    records = _wake_average(collections, end, period)
    title = f"Mean: $t = {end - period:.2g}$--${end:.2g}$ s"
    records.sort(key=lambda row: row[0][0] @ direction)
    axial = [field @ direction / speed for _, field in records]
    vertical = [field[:, 2] / speed for _, field in records]
    sequential = plt.get_cmap(_theme.COLORMAPS["velocity"])
    diverging = plt.get_cmap(_theme.COLORMAPS["vorticity"])
    if field_name == "streamwise":
        values, cmap, label = (axial, sequential, "$u_{\\parallel}/U_\\infty$")
    elif field_name == "vertical":
        values, cmap, label = (vertical, diverging, "$u_z/U_\\infty$")
    limits = (min((v.min() for v in values)), max((v.max() for v in values)))
    if field_name == "vertical":
        limit = max(abs(limits[0]), abs(limits[1]), 1e-08)
        limits = (-limit, limit)
    elif limits[1] - limits[0] < 1e-08:
        limits = (limits[0] - 1e-08, limits[1] + 1e-08)
    fig, axes = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(12.5 * _theme.CM, 7.6 * _theme.CM)
    )
    _theme.centered_subplots_adjust(fig, outer=0.097, bottom=0.4, top=0.83, wspace=0.13)
    for ax, (points, _), field in zip(axes, records, values, strict=True):
        artist = ax.tricontourf(
            points[:, 1], points[:, 2], field, levels=np.linspace(*limits, 25), cmap=cmap
        )
        ax.set(xlabel="$y$ [m]", title=f"$x={points[0, 0]:.2g}$ m")
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-1, 0])
    axes[0].set_ylabel("$z$ [m]")
    outer = 0.097
    panel_width_cm = axes[0].get_position().width * 12.5
    aspect = np.ptp(records[0][0][:, 2]) / np.ptp(records[0][0][:, 1])
    panel_height_cm = panel_width_cm * aspect
    height_cm = panel_height_cm + 3.5
    fig.set_size_inches(12.5 * _theme.CM, height_cm * _theme.CM, forward=False)
    fig.subplots_adjust(bottom=2.8 / height_cm, top=1 - 0.58 / height_cm)
    for ax in axes:
        ax.set_aspect("equal")
    cax = fig.add_axes([outer, 1.3 / height_cm, 1 - 2 * outer, 0.22 / height_cm])
    fig.colorbar(
        artist,
        cax=cax,
        orientation="horizontal",
        ticks=np.linspace(*limits, 3),
        format="%.2g",
        label=label,
    )
    print(title)
    _save_figure(fig, destination / f"delta_wing_wake_{field_name}.png", figure_format)
