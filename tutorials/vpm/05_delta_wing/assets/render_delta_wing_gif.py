#!/usr/bin/env python3
"""Render a 30 fps GIF from coupled VPM+VLM backup frames.

The VPM-owned backup is the source of both the particle restart state and the
attached VLM surface state. The renderer selects the nearest accepted backup
for each physical 1/30 s target; it never interpolates geometry or loads. The
selected backup timestamps are written beside the GIF for auditability. Repeat
``--solution`` in sparse-to-dense order for an explicit dense source; sparse
sources are rejected before any GIF is written. Without explicit directories,
the case-relative accepted-lineage manifest records the canonical source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PolyCollection

if not __package__:
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

from ._delta_wing_plots import (
    LINEAGE_MANIFEST,
    _theme,
    _finite_time,
    _integer_step,
    _validate_selected_clock,
    load_animation_lineage,
)


CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
DEFAULT_OUTPUT = CASE_DIR / "assets" / "delta_wing_30fps.gif"
GIF_FPS = 30
MAX_NATIVE_GAP = 1.0 / GIF_FPS
SERIALIZATION_RTOL = 5.0e-6
SERIALIZATION_ATOL = 1.0e-7
CLOCK_ATOL = 1.0e-12

# Fixed screen-space projection for the moving panels. A plain x-z view is
# nearly edge-on for this geometry because the wing span is mostly y. This
# oblique projection retains chord (x), span (y), and heave/pitch (z) in one
# consistent 2-D view without inventing intermediate solver states.
OBLIQUE_PROJECTION = np.array([[1.0, 0.28, 0.0], [0.0, 0.72, 1.0]])


def oblique_projection(points: np.ndarray) -> np.ndarray:
    """Project native x/y/z points to fixed oblique screen coordinates."""
    return np.asarray(points) @ OBLIQUE_PROJECTION.T


def _backup_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid coupled backup name: {path.name}") from error


def _read_backup_clock(path: Path) -> tuple[int, float]:
    with h5py.File(path, "r") as archive:
        if "solver" not in archive or "vlm" not in archive["solver"]:
            raise ValueError(f"{path.name}: missing coupled solver/vlm state")
        solver = archive["solver"]
        try:
            step = _integer_step(solver.attrs["step"], f"{path}: solver step")
            time = _finite_time(solver.attrs["time"], f"{path}: solver time")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: missing coupled solver clock") from error
        vlm = solver["vlm"]
        required = {"panel_corner_position", "circulation", "panel_force"}
        missing = required - set(vlm)
        if missing:
            raise ValueError(f"{path.name}: missing coupled VLM arrays {sorted(missing)}")
    if step < 0:
        raise ValueError(f"{path.name}: non-finite or invalid coupled solver clock")
    return step, time


def _compare_serialized_values(label: str, left, right) -> None:
    """Reject a conflicting common HDF5 value before handoff deduplication."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"overlapping coupled backups conflict in {label}: "
            f"shapes {left_array.shape} and {right_array.shape}"
        )
    numeric = left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc"
    if numeric:
        if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
            raise ValueError(f"overlapping coupled backups contain non-finite {label}")
        if left_array.dtype.kind in "biu" or right_array.dtype.kind in "biu":
            # IDs, counts and flags are schema values, not f32 payloads.
            equal = np.array_equal(left_array, right_array)
        else:
            equal = np.allclose(
                left_array,
                right_array,
                rtol=SERIALIZATION_RTOL,
                atol=SERIALIZATION_ATOL,
                equal_nan=False,
            )
    else:
        equal = np.array_equal(left_array, right_array)
    if not equal:
        raise ValueError(
            f"overlapping coupled backups conflict in {label} "
            f"(serialization tolerance rtol={SERIALIZATION_RTOL:g}, "
            f"atol={SERIALIZATION_ATOL:g})"
        )


def _compare_common_hdf5_datasets(left, right, prefix: str) -> None:
    """Compare common native datasets while allowing additive new fields."""
    for name in sorted(set(left) & set(right)):
        left_item = left[name]
        right_item = right[name]
        label = f"{prefix}/{name}"
        if isinstance(left_item, h5py.Dataset) != isinstance(right_item, h5py.Dataset):
            raise ValueError(
                f"overlapping coupled backups conflict at {label}: object type differs"
            )
        if isinstance(left_item, h5py.Dataset):
            _compare_serialized_values(label, left_item[()], right_item[()])
        else:
            _compare_common_hdf5_datasets(left_item, right_item, label)


def _compare_overlapping_backups(first: Path, second: Path) -> None:
    """Check a repeated accepted state before choosing the dense source row."""
    with h5py.File(first, "r") as left_archive, h5py.File(second, "r") as right_archive:
        left_solver = left_archive["solver"]
        right_solver = right_archive["solver"]
        required_solver_attrs = {
            "backup_format_version",
            "n_particles_total",
            "numerical_configuration",
            "numerical_configuration_sha256",
            "freestream_velocity",
            "step",
            "time",
            "time_step_size",
        }
        missing = sorted(
            name
            for name in required_solver_attrs
            if name not in left_solver.attrs or name not in right_solver.attrs
        )
        if missing:
            raise ValueError(
                f"overlapping coupled backups missing required solver fields {missing}"
            )
        for name in sorted(set(left_solver.attrs) & set(right_solver.attrs)):
            if name in {"step", "time", "time_step_size"}:
                # The accepted clock is a double-precision contract, not a
                # relaxed comparison of physical f32 payloads.
                if name == "step":
                    equal = int(left_solver.attrs[name]) == int(right_solver.attrs[name])
                else:
                    equal = bool(
                        np.isclose(
                            float(left_solver.attrs[name]),
                            float(right_solver.attrs[name]),
                            rtol=0.0,
                            atol=CLOCK_ATOL,
                        )
                    )
                if not equal:
                    raise ValueError(
                        f"overlapping coupled backups conflict in solver attribute {name}"
                    )
            else:
                _compare_serialized_values(
                    f"solver attribute {name}", left_solver.attrs[name], right_solver.attrs[name]
                )

        left_vlm = left_solver["vlm"]
        right_vlm = right_solver["vlm"]
        for name in ("coupled_mode", "reference_speed", "reference_velocity", "solved", "time"):
            if name not in left_vlm.attrs or name not in right_vlm.attrs:
                raise ValueError(f"overlapping coupled backups missing VLM attribute {name}")
        left_version = int(left_vlm.attrs.get("version", -1))
        right_version = int(right_vlm.attrs.get("version", -1))
        if left_version != right_version and {left_version, right_version} != {6, 7}:
            raise ValueError(
                f"overlapping coupled backups conflict in VLM version: "
                f"{left_version} versus {right_version}"
            )
        for name in sorted(set(left_vlm.attrs) & set(right_vlm.attrs)):
            if name in {"identity", "version"}:
                continue
            if name == "time":
                if not np.isclose(
                    float(left_vlm.attrs[name]),
                    float(right_vlm.attrs[name]),
                    rtol=0.0,
                    atol=CLOCK_ATOL,
                ):
                    raise ValueError(
                        "overlapping coupled backups conflict in solver/vlm attribute time"
                    )
            else:
                _compare_serialized_values(
                    f"solver/vlm attribute {name}", left_vlm.attrs[name], right_vlm.attrs[name]
                )
        left_physics_identity = left_vlm.attrs.get("physics_identity")
        right_physics_identity = right_vlm.attrs.get("physics_identity")
        if left_physics_identity is not None and right_physics_identity is not None:
            _compare_serialized_values(
                "solver/vlm attribute physics_identity",
                left_physics_identity,
                right_physics_identity,
            )
        left_identity = left_vlm.attrs.get("identity")
        right_identity = right_vlm.attrs.get("identity")
        if left_identity is None or right_identity is None:
            if left_version == right_version and left_identity != right_identity:
                raise ValueError("overlapping coupled backups have incomplete VLM identity")
        elif left_version == right_version:
            _compare_serialized_values(
                "solver/vlm attribute identity", left_identity, right_identity
            )
        # A v6 -> v7 continuation may legitimately change the legacy full
        # identity as output-only fields migrate. The common physics identity,
        # numerical identity and clocks above still have to agree.

        _compare_common_hdf5_datasets(
            left_archive["particles"], right_archive["particles"], "particles"
        )
        _compare_common_hdf5_datasets(left_vlm, right_vlm, "solver/vlm")


def _solution_directories(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    directories = [Path(directory) for directory in value]
    if not directories:
        raise ValueError("at least one solution directory is required")
    return directories


def coupled_frames(
    solution_dir: Path | list[Path] | None = None,
    lineage_manifest: Path = LINEAGE_MANIFEST,
) -> list[tuple[float, Path, str]]:
    """Return ordered coupled backups from one or more named source segments."""
    lineage = load_animation_lineage(lineage_manifest) if solution_dir is None else None
    if lineage is None:
        sources = [(directory, None) for directory in _solution_directories(solution_dir)]
    else:
        sources = [(segment["solution_path"], segment) for segment in lineage]
    by_step: dict[int, tuple[float, Path, str]] = {}
    for directory, segment_info in sources:
        paths = sorted((directory / "vpm").glob("vpm_*.h5"), key=_backup_step)
        if segment_info is not None:
            interval = segment_info["accepted_interval"]
            paths = [
                path
                for path in paths
                if _backup_step(path) >= interval["first_step"]
                and (interval["last_step"] is None or _backup_step(path) <= interval["last_step"])
            ]
        if not paths:
            raise FileNotFoundError(f"no coupled VPM backups found under {directory}")
        segment = (
            segment_info["id"] if segment_info is not None else directory.name or str(directory)
        )
        for path in paths:
            step, time = _read_backup_clock(path)
            if segment_info is not None:
                interval = segment_info["accepted_interval"]
                _validate_selected_clock(step, time, interval, f"{segment} {path.name}")
            if step in by_step:
                previous_time, previous_path, _ = by_step[step]
                if not np.isclose(previous_time, time, rtol=0.0, atol=CLOCK_ATOL):
                    raise ValueError(f"handoff step {step} has inconsistent source timestamps")
                _compare_overlapping_backups(previous_path, path)
            # If a continuation repeats the handoff backup, its later segment
            # owns the merged record while both source directories remain in
            # the output manifest.
            by_step[step] = (time, path, segment)
    records = [by_step[step] for step in sorted(by_step)]
    steps = np.asarray([_backup_step(path) for _, path, _ in records])
    times = np.asarray([time for time, _, _ in records])
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("coupled VPM backup steps or timestamps are unordered")
    return records


def select_frames(records: list[tuple[float, Path, str]], fps: int = GIF_FPS):
    if fps != GIF_FPS:
        raise ValueError(f"GIF output is fixed at {GIF_FPS} fps")
    source_times = np.asarray([time for time, _, _ in records])
    count = max(1, int(round((source_times[-1] - source_times[0]) * fps)) + 1)
    targets = np.linspace(source_times[0], source_times[-1], count)
    indexes = np.searchsorted(source_times, targets).clip(0, len(source_times) - 1)
    left = np.maximum(indexes - 1, 0)
    choose_left = np.abs(source_times[left] - targets) < np.abs(source_times[indexes] - targets)
    indexes[choose_left] = left[choose_left]
    # Keep every presentation target. A sparse native frame is held across
    # several targets; dropping those targets would make the GIF far shorter
    # than its physical source interval while still claiming the requested fps.
    return targets, [records[int(index)] for index in indexes]


def _require_dense_native_source(records: list[tuple[float, Path, str]]) -> float:
    """Reject sparse native sources before a public dense animation is written."""
    if len(records) < 2:
        raise ValueError("dense animation requires at least two native backups")
    source_times = np.asarray([time for time, _, _ in records], dtype=float)
    maximum_gap = float(np.max(np.diff(source_times)))
    if maximum_gap > MAX_NATIVE_GAP + CLOCK_ATOL:
        raise ValueError(
            "dense animation requires native backup gaps at most "
            f"1/{GIF_FPS} s (observed {maximum_gap:g} s)"
        )
    return maximum_gap


def _require_unique_presentation_sources(
    selected: list[tuple[float, Path, str]],
) -> None:
    """Reject a presentation sequence that would hold one native state twice."""
    source_paths = [path.resolve() for _, path, _ in selected]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("dense animation would repeat a native state across presentation frames")


def _read_vlm_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as archive:
        vlm = archive["solver/vlm"]
        corners = np.asarray(vlm["panel_corner_position"], dtype=float)
        circulation = np.asarray(vlm["circulation"], dtype=float)
        panel_force = np.asarray(vlm["panel_force"], dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 3):
        raise ValueError(f"{path.name}: invalid coupled VLM panel corners")
    if circulation.shape != (len(corners),) or panel_force.shape != (len(corners), 3):
        raise ValueError(f"{path.name}: inconsistent coupled VLM panel arrays")
    if (
        not np.isfinite(corners).all()
        or not np.isfinite(circulation).all()
        or not np.isfinite(panel_force).all()
    ):
        raise ValueError(f"{path.name}: non-finite coupled VLM geometry or loads")
    return corners, circulation


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(CASE_DIR))
    except ValueError:
        return str(path)


def _source_segments(records: list[tuple[float, Path, str]]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for time, path, segment in records:
        if not segments or segments[-1]["namespace"] != segment:
            segments.append(
                {
                    "namespace": segment,
                    "directory": _display_path(path.parent),
                    "first_step": _backup_step(path),
                    "last_step": _backup_step(path),
                    "first_time": float(time),
                    "last_time": float(time),
                    "frame_count": 1,
                }
            )
        else:
            segments[-1]["last_step"] = _backup_step(path)
            segments[-1]["last_time"] = float(time)
            segments[-1]["frame_count"] = int(segments[-1]["frame_count"]) + 1
    return segments


def _segment_transitions(records: list[tuple[float, Path, str]]) -> list[dict[str, object]]:
    transitions = []
    for previous, current in zip(records[:-1], records[1:], strict=True):
        if previous[2] == current[2]:
            continue
        transitions.append(
            {
                "from_namespace": previous[2],
                "to_namespace": current[2],
                "from_step": _backup_step(previous[1]),
                "to_step": _backup_step(current[1]),
                "accepted_step_gap": _backup_step(current[1]) - _backup_step(previous[1]),
                "time_gap": float(current[0] - previous[0]),
            }
        )
    return transitions


def render(
    output: Path,
    solution_dir: Path | list[Path] | None = None,
    fps: int = GIF_FPS,
    lineage_manifest: Path = LINEAGE_MANIFEST,
) -> None:
    if fps != GIF_FPS:
        raise ValueError(f"GIF output is fixed at {GIF_FPS} fps")
    lineage = load_animation_lineage(lineage_manifest) if solution_dir is None else None
    records = coupled_frames(solution_dir, lineage_manifest)
    maximum_native_gap = _require_dense_native_source(records)
    target_times, selected = select_frames(records, fps)
    _require_unique_presentation_sources(selected)
    native_frames = [
        (time, path, segment, *_read_vlm_frame(path)) for time, path, segment in selected
    ]

    vertices = np.concatenate(
        [oblique_projection(corners) for _, _, _, corners, _ in native_frames]
    ).reshape(-1, 2)
    circulation = np.concatenate([values for _, _, _, _, values in native_frames])
    horizontal_limits = (vertices[:, 0].min() - 0.15, vertices[:, 0].max() + 0.15)
    vertical_limits = (vertices[:, 1].min() - 0.15, vertices[:, 1].max() + 0.15)
    scale = max(float(np.max(np.abs(circulation))), 1e-12)
    norm = Normalize(-scale, scale)
    _theme.set_thesis_style()
    cmap = LinearSegmentedColormap.from_list(
        "thesis_signed", [_theme.COLORS["TUDcyan"], "white", _theme.COLORS["VPMpurple"]]
    )
    frames = []
    for target_time, (source_time, _, source_segment), (_, path, segment, corners, values) in zip(
        target_times, selected, native_frames, strict=True
    ):
        fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.5 * _theme.CM), dpi=160)
        polygons = oblique_projection(corners)
        ax.add_collection(
            PolyCollection(
                polygons,
                array=values,
                cmap=cmap,
                norm=norm,
                edgecolors=_theme.COLORS["DarkText"],
                linewidths=0.15,
            )
        )
        ax.set(
            xlim=horizontal_limits,
            ylim=vertical_limits,
            xlabel="$s$ [m]",
            ylabel="$h$ [m]",
            title=f"$t = {source_time:.3f}$ s",
        )
        ax.grid(True, color=_theme.COLORS["LightBG"], linewidth=0.5)
        ax.locator_params(axis="y", nbins=3)
        _theme.centered_subplots_adjust(fig, outer=0.15, bottom=0.30, top=0.87)
        _theme.fit_thesis_y_label_margins(fig, (ax,))
        outer = ax.get_position().x0
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
            format="%.2f",
            label=r"$\Gamma$ [m$^2$/s]",
        )
        _theme.validate_thesis_figure(fig, (ax, cax))
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(pixels))
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    # GIF stores durations in centiseconds. The repeating 30/30/40 ms
    # pattern has an exact 100 ms / 3-frame average, i.e. 30 fps.
    durations = [30 if index % 3 < 2 else 40 for index in range(len(frames))]
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2
    )
    manifest = output.with_suffix(".json")
    sparse_hold = len({path for _, path, _ in selected}) < len(selected)
    manifest.write_text(
        json.dumps(
            {
                "gif": _display_path(output),
                "fps": fps,
                "frame_duration_pattern_ms": [30, 30, 40],
                "source_format": "coupled_vpm_hdf5",
                "source_selection": (
                    "finalized_dense_lineage" if lineage is not None else "explicit_directories"
                ),
                "source_lineage_manifest": (
                    _display_path(Path(lineage_manifest).resolve()) if lineage is not None else None
                ),
                "source_lineage_segments": (
                    [segment["id"] for segment in lineage] if lineage is not None else None
                ),
                "overlap_validation": {
                    "clock_atol": CLOCK_ATOL,
                    "serialization_rtol": SERIALIZATION_RTOL,
                    "serialization_atol": SERIALIZATION_ATOL,
                    "additive_fields_allowed": True,
                },
                "sampling_mode": "nearest_native_hold" if sparse_hold else "nearest_native_backup",
                "sparse_hold": sparse_hold,
                "maximum_native_gap_seconds": maximum_native_gap,
                "presentation_repeated_native_state": sparse_hold,
                "native_source_count": len(records),
                "native_source_backups": [_display_path(path) for _, path, _ in records],
                "native_source_timestamps": [float(time) for time, _, _ in records],
                "source_segments": _source_segments(records),
                "segment_transitions": _segment_transitions(records),
                "physical_start_time": float(records[0][0]),
                "physical_end_time": float(records[-1][0]),
                "frame_count": len(frames),
                "source_backups": [_display_path(path) for _, path, _ in selected],
                "source_backup_segments": [segment for _, _, segment in selected],
                "accepted_source_timestamps": [
                    float(source_time) for source_time, _, _ in selected
                ],
                "target_timestamps": [float(target_time) for target_time in target_times],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(frames)} frames at {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="dense solution directory; repeat in sparse-to-dense order for a continuation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=GIF_FPS)
    args = parser.parse_args()
    if args.fps != GIF_FPS:
        parser.error(f"--fps must be exactly {GIF_FPS}")
    render(args.output, args.solution, args.fps)


if __name__ == "__main__":
    main()
