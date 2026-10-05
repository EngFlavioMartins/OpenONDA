"""Forces, measured heave cycles, period-mean wakes and native particle scenes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.signal import find_peaks

from openonda import plotting as _theme
from openonda.results import read_csv_table, read_json, read_pvd_frames
from openonda.scenes import export_animation, figure_frame
from source.solvers.vpm.io.postprocess import backup_frames, surface_mean, vlm_surface

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples/delta_wing"
FIGURES_DIR = CASE_DIR / "figures"
DEFAULT_GIF_OUTPUT = FIGURES_DIR / "delta_wing_30fps.gif"
GIF_FPS = 30
OBLIQUE_PROJECTION = np.array([[1.0, 0.28, 0.0], [0.0, 0.72, 1.0]])


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


def _save_figure(fig, axes, path, figure_format, *, fit=True):
    return _theme.export_figure(fig, path, figure_format=figure_format)


def _wake_average(collections, end, period):
    """Average one measured heave cycle on each native wake plane."""
    return [
        surface_mean(
            [(time, path) for time, path, _ in frames], end - period, end, field="velocity"
        )
        for frames in collections
    ]


def oblique_projection(points: np.ndarray) -> np.ndarray:
    """Project native x/y/z points to fixed oblique screen coordinates."""
    return np.asarray(points) @ OBLIQUE_PROJECTION.T


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
    _save_figure(
        fig,
        (*axes, cax),
        destination / f"delta_wing_wake_{field_name}.png",
        figure_format,
        fit=False,
    )


def render(output: Path, solution_dir=None, fps: int = GIF_FPS) -> None:
    paths = _directories(solution_dir) if solution_dir is not None else [SOLUTION_DIR]
    records = sorted(
        (
            (time, path, directory.name)
            for directory in paths
            for time, path in backup_frames(directory)
        )
    )
    native_times = np.asarray([time for time, _, _ in records])
    target_times = np.arange(native_times[0], native_times[-1] + 0.5 / fps, 1.0 / fps)
    selected = [records[np.argmin(abs(native_times - time))] for time in target_times]
    native_frames = [(time, path, segment, *vlm_surface(path)) for time, path, segment in selected]
    vertices = np.concatenate(
        [oblique_projection(corners) for _, _, _, corners, _ in native_frames]
    ).reshape(-1, 2)
    circulation = np.concatenate([values for _, _, _, _, values in native_frames])
    horizontal_limits = (vertices[:, 0].min() - 0.15, vertices[:, 0].max() + 0.15)
    vertical_limits = (vertices[:, 1].min() - 0.15, vertices[:, 1].max() + 0.15)
    scale = max(float(np.max(np.abs(circulation))), 1e-12)
    norm = Normalize(-scale, scale)
    _theme.set_thesis_style()
    cmap = plt.get_cmap(_theme.COLORMAPS["vorticity"])
    frames = []
    for _target_time, (source_time, _, _source_segment), (
        _,
        _path,
        _segment,
        corners,
        values,
    ) in zip(target_times, selected, native_frames, strict=True):
        fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.5 * _theme.CM), dpi=160)
        polygons = oblique_projection(corners)
        ax.add_collection(
            PolyCollection(
                polygons,
                array=values,
                cmap=cmap,
                norm=norm,
                edgecolors=_theme.COLORS["text"],
                linewidths=0.6,
            )
        )
        ax.set(
            xlim=horizontal_limits,
            ylim=vertical_limits,
            xlabel="$s$ [m]",
            ylabel="$h$ [m]",
            title=f"$t = {source_time:.2g}$ s",
        )
        ax.grid(False)
        ax.locator_params(axis="y", nbins=3)
        _theme.centered_subplots_adjust(fig, outer=0.15, bottom=0.3, top=0.87)
        outer = 0.115
        _theme.centered_subplots_adjust(fig, outer=outer)
        height_cm = (1 - 2 * outer) * 12.5 * np.ptp(vertical_limits) / np.ptp(
            horizontal_limits
        ) + 3.8
        fig.set_size_inches(12.5 * _theme.CM, height_cm * _theme.CM, forward=False)
        fig.subplots_adjust(bottom=2.8 / height_cm, top=1 - 1.0 / height_cm)
        ax.set_aspect("equal")
        cax = fig.add_axes([outer, 1.3 / height_cm, 1 - 2 * outer, 0.22 / height_cm])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            ticks=[-scale, 0, scale],
            format="%.2g",
            label="$\\Gamma$ [m$^2$/s]",
        )
        frames.append(figure_frame(fig))
        plt.close(fig)
    export_animation(frames, output, fps=fps)
    print(f"wrote {output} ({len(frames)} frames at {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render-gif",))
    parser.add_argument("--solution", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_GIF_OUTPUT)
    parser.add_argument("--fps", type=int, default=GIF_FPS)
    args = parser.parse_args()
    render(args.output, args.solution, args.fps)


if __name__ == "__main__":
    main()
