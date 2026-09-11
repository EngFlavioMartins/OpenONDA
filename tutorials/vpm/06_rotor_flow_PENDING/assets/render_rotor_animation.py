#!/usr/bin/env python3
"""Render coupled VPM+VLM backups alongside recorded wake-plane fields.

The rotor surface in each animation frame is read from the VPM-owned HDF5
backup, which is also the numerical restart state. A nearest native wake-plane
sample is paired for context. Neither source is interpolated or reconstructed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
from defusedxml import ElementTree
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CASE_DIR / "assets" / "animation" / "rotor_30fps.gif"


def _run_directories() -> tuple[Path, Path]:
    tag = os.environ.get("ROTOR_OUTPUT_TAG", "completion")
    solution = CASE_DIR / "solution" / tag if tag else CASE_DIR / "solution"
    samples = CASE_DIR / "samples" / "rotor" / tag if tag else CASE_DIR / "samples" / "rotor"
    return solution, samples


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
            step = int(solver.attrs["step"])
            time = float(solver.attrs["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: missing coupled solver clock") from error
        vlm = solver["vlm"]
        required = {"panel_corner_position", "circulation"}
        missing = required - set(vlm)
        if missing:
            raise ValueError(f"{path.name}: missing coupled VLM arrays {sorted(missing)}")
    if step < 0 or not np.isfinite(time):
        raise ValueError(f"{path.name}: non-finite or invalid coupled solver clock")
    return step, time


def _coupled_frames(solution_dir: Path) -> list[tuple[float, Path]]:
    """Return strictly ordered VPM-owned backups containing attached VLM state."""
    paths = sorted(solution_dir.glob("vpm_*.h5"), key=_backup_step)
    if not paths:
        raise FileNotFoundError(f"no coupled VPM backups found under {solution_dir}")
    records = [(time, path) for path in paths for _, time in [_read_backup_clock(path)]]
    steps = np.asarray([_backup_step(path) for _, path in records])
    times = np.asarray([time for time, _ in records])
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("coupled VPM backup steps or timestamps are unordered")
    return records


def _pvd_frames(path: Path) -> list[tuple[float, Path]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ElementTree.parse(path)
    frames = [
        (float(item.attrib["timestep"]), path.parent / item.attrib["file"])
        for item in root.findall(".//DataSet")
    ]
    if not frames or not np.isfinite([time for time, _ in frames]).all():
        raise ValueError(f"{path}: no finite native frames")
    times = np.asarray([time for time, _ in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: frame times are not strictly increasing")
    return frames


def _nearest_frame(frames, time: float) -> tuple[float, Path]:
    index = int(np.abs(np.asarray([item[0] for item in frames]) - time).argmin())
    return frames[index]


def _panel_centres(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as archive:
        vlm = archive["solver/vlm"]
        corners = np.asarray(vlm["panel_corner_position"], dtype=float)
        circulation = np.asarray(vlm["circulation"], dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 3):
        raise ValueError(f"{path.name}: invalid coupled VLM panel corners")
    centres = corners.mean(axis=1)
    if circulation.shape != (len(centres),):
        raise ValueError(f"{path.name}: inconsistent coupled VLM panel arrays")
    if not np.isfinite(centres).all() or not np.isfinite(circulation).all():
        raise ValueError(f"{path.name}: non-finite coupled VLM geometry or circulation")
    return centres, circulation


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(CASE_DIR))
    except ValueError:
        return str(path)


def render(*, output: Path, fps: float = 30.0, max_frames: int | None = None) -> Path:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if fps > 100.0:
        raise ValueError("GIF playback cannot represent more than 100 fps")
    solution_dir, samples_dir = _run_directories()
    vlm_frames = _coupled_frames(solution_dir)
    plane_frames = _pvd_frames(samples_dir / "wake_1D.pvd")
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        vlm_frames = vlm_frames[:max_frames]
    physical_span = vlm_frames[-1][0] - vlm_frames[0][0]
    gif_span = len(vlm_frames) / fps
    playback = physical_span / gif_span if gif_span > 0 else np.nan
    output.parent.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    native_frames = []
    for time, backup_path in vlm_frames:
        centres, circulation = _panel_centres(backup_path)
        native_frames.append((time, backup_path, centres, circulation))
    circulation_all = np.concatenate([circulation for _, _, _, circulation in native_frames])
    circulation_norm = Normalize(
        vmin=float(circulation_all.min()), vmax=float(circulation_all.max())
    )
    if circulation_norm.vmin == circulation_norm.vmax:
        circulation_norm = Normalize(
            vmin=circulation_norm.vmin - 1.0, vmax=circulation_norm.vmax + 1.0
        )
    plane_cache = {}
    for time, _ in vlm_frames:
        plane_time, plane_path = _nearest_frame(plane_frames, time)
        if plane_path not in plane_cache:
            plane = pv.read(plane_path)
            values = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
            plane_cache[plane_path] = (plane_time, np.asarray(plane.points), values)
    wake_points = next(iter(plane_cache.values()))[1]
    vorticity_max = max(values.max() for _, _, values in plane_cache.values())
    vorticity_norm = Normalize(vmin=0.0, vmax=max(float(vorticity_max), 1.0e-12))
    panel_limits = (
        float(np.min([centres[:, 1].min() for _, _, centres, _ in native_frames])),
        float(np.max([centres[:, 1].max() for _, _, centres, _ in native_frames])),
        float(np.min([centres[:, 2].min() for _, _, centres, _ in native_frames])),
        float(np.max([centres[:, 2].max() for _, _, centres, _ in native_frames])),
    )
    panel_margin = max(panel_limits[1] - panel_limits[0], panel_limits[3] - panel_limits[2]) * 0.05
    wake_limits = (
        float(wake_points[:, 1].min()),
        float(wake_points[:, 1].max()),
        float(wake_points[:, 2].min()),
        float(wake_points[:, 2].max()),
    )
    native_frame_count = len(native_frames)
    selected_plane_paths = []
    for frame_index, (time, backup_path, centres, circulation) in enumerate(native_frames):
        plane_time, plane_path = _nearest_frame(plane_frames, time)
        selected_plane_paths.append(plane_path)
        _, points, vorticity = plane_cache[plane_path]
        if len(points) != len(vorticity) or not np.isfinite(vorticity).all():
            raise ValueError(f"{plane_path}: invalid native vorticity field")

        for axis in axes:
            axis.clear()
        axes[0].scatter(
            centres[:, 1], centres[:, 2], c=circulation, s=7, cmap="coolwarm", norm=circulation_norm
        )
        axes[0].set(xlabel="rotor y [m]", ylabel="rotor z [m]", title="coupled VPM+VLM rotor disk")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(panel_limits[0] - panel_margin, panel_limits[1] + panel_margin)
        axes[0].set_ylim(panel_limits[2] - panel_margin, panel_limits[3] + panel_margin)
        axes[1].scatter(
            points[::4, 1],
            points[::4, 2],
            c=vorticity[::4],
            s=1.2,
            cmap="magma",
            norm=vorticity_norm,
        )
        axes[1].set(
            xlabel="wake-plane y [m]",
            ylabel="wake-plane z [m]",
            title=f"native wake_1D field (t={plane_time:.3f} s)",
        )
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlim(wake_limits[0], wake_limits[1])
        axes[1].set_ylim(wake_limits[2], wake_limits[3])
        figure.suptitle(
            f"rotor coupled backup frames | accepted t={time:.3f} s | frame {frame_index + 1}/{native_frame_count} | "
            f"GIF {fps:g} fps | playback {playback:.2f}x",
            fontsize=12,
        )
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        images.append(Image.fromarray(rgba[:, :, :3].copy()))
    plt.close(figure)
    # GIF stores frame delays in 10 ms units. Round cumulative timestamps so
    # 40 ms frames are distributed throughout the sequence instead of
    # creating a front-loaded playback-speed jump.
    timestamps = np.rint(np.arange(len(images) + 1) * 100.0 / fps).astype(int)
    durations = (10 * np.diff(timestamps)).tolist()
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    output_manifest = output.with_suffix(".json")
    output_manifest.write_text(
        json.dumps(
            {
                "gif": _display_path(output),
                "fps": fps,
                "source_format": "coupled_vpm_hdf5",
                "source_backups": [_display_path(path) for _, path in vlm_frames],
                "source_backup_timestamps": [float(time) for time, _ in vlm_frames],
                "wake_plane_source": _display_path(samples_dir / "wake_1D.pvd"),
                "wake_plane_frames": [_display_path(path) for path in selected_plane_paths],
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    print(render(output=args.output, fps=args.fps, max_frames=args.max_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
