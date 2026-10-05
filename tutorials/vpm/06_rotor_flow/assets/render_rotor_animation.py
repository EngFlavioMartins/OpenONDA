"""Render coupled VPM+VLM backups alongside recorded wake-plane fields.

The rotor surface in each animation frame is read from the VPM-owned HDF5
backup, which is also the numerical restart state. A nearest native wake-plane
sample is paired for context. Neither source is interpolated or reconstructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from openonda import plotting as theme
from openonda.results import read_pvd_frames, write_text
from openonda.scenes import export_animation, figure_frame
from source.solvers.vpm.io.postprocess import coupled_frames, vlm_surface

CASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CASE_DIR / "figures" / "rotor_30fps.gif"


def _nearest_frame(frames, time: float) -> tuple[float, Path]:
    index = int(np.abs(np.asarray([item[0] for item in frames]) - time).argmin())
    return frames[index]


def render(*, output: Path, fps: float = 30.0, max_frames: int | None = None) -> Path:
    solution_dir, samples_dir = (CASE_DIR / "solution", CASE_DIR / "samples/rotor")
    vlm_frames = coupled_frames(solution_dir)
    plane_frames = [
        (time, path)
        for time, path in read_pvd_frames(samples_dir / "wake_1D.pvd")
        if time <= vlm_frames[-1][0]
    ]
    if max_frames is not None:
        vlm_frames = vlm_frames[:max_frames]
    physical_span = vlm_frames[-1][0] - vlm_frames[0][0]
    gif_span = len(vlm_frames) / fps
    playback = physical_span / gif_span if gif_span > 0 else np.nan
    images = []
    theme.set_thesis_style()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100, constrained_layout=True)
    circulation_cmap = plt.get_cmap(theme.COLORMAPS["vorticity"])
    vorticity_cmap = plt.get_cmap(theme.COLORMAPS["vorticity_magnitude"])
    native_frames = []
    for time, backup_path in vlm_frames:
        corners, circulation = vlm_surface(backup_path)
        centres = corners.mean(axis=1)
        native_frames.append((time, backup_path, centres, circulation))
    circulation_all = np.concatenate([circulation for _, _, _, circulation in native_frames])
    circulation_scale = max(float(np.abs(circulation_all).max()), 1e-12)
    circulation_norm = Normalize(vmin=-circulation_scale, vmax=circulation_scale)
    plane_fields = []
    for time, _ in vlm_frames:
        plane_time, plane_path = _nearest_frame(plane_frames, time)
        plane = pv.read(plane_path)
        values = np.linalg.norm(np.asarray(plane["vorticity"]), axis=1)
        plane_fields.append((plane_time, np.asarray(plane.points), values))
    wake_points = plane_fields[0][1]
    vorticity_max = max((values.max() for _, _, values in plane_fields))
    vorticity_norm = Normalize(vmin=0.0, vmax=max(float(vorticity_max), 1e-12))
    figure.colorbar(
        ScalarMappable(norm=circulation_norm, cmap=circulation_cmap),
        ax=axes[0],
        location="bottom",
        shrink=0.8,
        pad=0.13,
        label="Signed panel circulation, $\\Gamma$ [m$^2$/s]",
    )
    figure.colorbar(
        ScalarMappable(norm=vorticity_norm, cmap=vorticity_cmap),
        ax=axes[1],
        location="bottom",
        shrink=0.8,
        pad=0.13,
        label="Vorticity magnitude, $|\\omega|$ [1/s]",
    )
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
    selected_plane_paths = []
    for frame_index, (time, backup_path, _centres, circulation) in enumerate(native_frames):
        plane_time, plane_path = _nearest_frame(plane_frames, time)
        selected_plane_paths.append(plane_path)
        _, points, vorticity = plane_fields[frame_index]
        for axis in axes:
            axis.clear()
        corners, _ = vlm_surface(backup_path)
        panels = PolyCollection(
            corners[:, :, 1:],
            array=circulation,
            cmap=circulation_cmap,
            norm=circulation_norm,
            edgecolors="none",
        )
        axes[0].add_collection(panels)
        axes[0].set(xlabel="y [m]", ylabel="z [m]", title="Native blade panels")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(panel_limits[0] - panel_margin, panel_limits[1] + panel_margin)
        axes[0].set_ylim(panel_limits[2] - panel_margin, panel_limits[3] + panel_margin)
        dimensions = pv.read(plane_path).dimensions[:2]
        axes[1].pcolormesh(
            points[:, 1].reshape(dimensions, order="F"),
            points[:, 2].reshape(dimensions, order="F"),
            vorticity.reshape(dimensions, order="F"),
            shading="nearest",
            cmap=vorticity_cmap,
            norm=vorticity_norm,
        )
        axes[1].set(xlabel="y [m]", ylabel="z [m]", title=f"1D wake plane, t={plane_time:.2g} s")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlim(wake_limits[0], wake_limits[1])
        axes[1].set_ylim(wake_limits[2], wake_limits[3])
        figure.suptitle(f"$t={time:.2g}$ s; {playback:.2g}x playback")
        images.append(figure_frame(figure))
    plt.close(figure)
    export_animation(images, output, fps=fps)
    output_manifest = output.with_suffix(".json")
    write_text(
        output_manifest,
        json.dumps(
            {
                "gif": str(output),
                "fps": fps,
                "source_format": "coupled_vpm_hdf5",
                "source_backups": [str(path) for _, path in vlm_frames],
                "source_backup_timestamps": [float(time) for time, _ in vlm_frames],
                "wake_plane_source": str(samples_dir / "wake_1D.pvd"),
                "wake_plane_frames": [str(path) for path in selected_plane_paths],
            },
            indent=2,
        )
        + "\n",
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
    main()
