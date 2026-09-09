"""Delta-wing figures from native force, motion and velocity samples."""

from pathlib import Path
import json

from defusedxml import ElementTree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.signal import find_peaks

from openonda import plotting as _theme

CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
FIGURES_DIR = CASE_DIR / "figures"
_COLORS = _theme.COLORS


def force_history(samples_dir=SAMPLES_DIR):
    return pd.read_csv(samples_dir / "vlm_surface_forces.csv")


def motion_period(data):
    """Measure the prescribed period from the solver's sampled heave velocity."""
    front = data[data.surface == "front_wing"].sort_values("time")
    peaks, _ = find_peaks(front.translation_velocity_z)
    if len(peaks) < 2:
        raise ValueError("At least two sampled heave-velocity peaks are needed to measure a period")
    return float(np.median(np.diff(front.time.to_numpy()[peaks])))


def last_cycles(data, period, count=3):
    """Yield complete cycles; a sample exactly on the endpoint belongs to the prior cycle."""
    times = np.sort(data.time.unique())
    cadence = np.median(np.diff(times))
    first = max(0, int(np.ceil((times[0] - cadence - 1e-9) / period)))
    last = int(np.floor((data.time.max() + 1e-9) / period))
    for cycle in range(max(first, last - count), last):
        rows = data[
            (data.time > cycle * period + 1e-9) & (data.time <= (cycle + 1) * period + 1e-9)
        ]
        yield cycle, (rows.time.to_numpy() - cycle * period) / period, rows


def plot_forces(samples_dir, figures_dir, figure_format="png"):
    _theme.set_thesis_style()
    data = force_history(samples_dir)
    period = motion_period(data)
    fig, rows = plt.subplots(
        4, 1, figsize=(12.5 * _theme.CM, 22 * _theme.CM), constrained_layout=True
    )
    axes = rows.reshape(2, 2)
    for surface, color, label in (
        ("front_wing", _COLORS["TUDcyan"], "Front"),
        ("rear_wing", _COLORS["VPMpurple"], "Rear"),
    ):
        rows = data[data.surface == surface]
        axes[0, 0].plot(rows.time, rows.force_z, color=color, label=label)
        axes[1, 0].plot(rows.time, rows.centroid_z, color=color)
        for i, (cycle, phase, tail) in enumerate(last_cycles(rows, period)):
            axes[0, 1].plot(
                phase,
                tail.force_z,
                color=color,
                ls=(":", "--", "-")[i],
                label=f"{label}, cycle {cycle + 1}",
            )
        axes[1, 1].plot(rows.time, -rows.power, color=color)
    axes[0, 0].set(xlabel="Time [s]", ylabel="Vertical force [N]")
    axes[1, 0].set(xlabel="Time [s]", ylabel="Sampled centroid z [m]")
    cycle_count = min(3, int(np.floor((data.time.max() + 1e-9) / period)))
    axes[0, 1].set(
        xlabel="Cycle phase",
        ylabel="Vertical force [N]",
        title=f"Complete cycles shown: {cycle_count}",
    )
    axes[1, 1].set(xlabel="Time [s]", ylabel="Motion input power [W]")
    axes[0, 0].legend()
    axes[0, 1].legend(ncol=2)
    for ax in axes.flat:
        ax.axhline(0, color="0.6", lw=0.4)
    _theme.save_fig(
        fig, figures_dir / "delta_wing_forces.png", figure_format=figure_format, bbox_inches=None
    )


def plot_circulation(samples_dir, figures_dir, figure_format="png"):
    _theme.set_thesis_style()
    data = pd.read_csv(samples_dir / "flow_integrals.csv")
    fig, ax = plt.subplots(figsize=_theme.figure_size("single_tall"), constrained_layout=True)
    ax.plot(data.time, data.vortex_strength_magnitude_sum, color=_COLORS["VPMpurple"])
    ax.set(
        xlabel="Time [s]",
        ylabel=r"$\sum_p |\boldsymbol{\alpha}_p|$ [m$^3$/s]",
        title="Wake vector-strength magnitude (not conserved)",
    )
    _theme.save_fig(
        fig,
        figures_dir / "delta_wing_circulation_history.png",
        figure_format=figure_format,
        bbox_inches=None,
    )


def plot_wake(samples_dir, figures_dir, figure_format="png"):
    _theme.set_thesis_style()
    metadata = json.loads((CASE_DIR / "solution/vpm_metadata.json").read_text())
    velocity = np.asarray(metadata["configuration"]["numerics"]["freestream_velocity"])
    speed = np.linalg.norm(velocity)
    direction = velocity / speed
    period = motion_period(force_history(samples_dir))
    planes = sorted(samples_dir.glob("wake_*span.pvd"))
    fig, axes = plt.subplots(
        len(planes),
        2,
        figsize=(12.5 * _theme.CM, 18 * _theme.CM),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.T
    records = []
    collections = [ElementTree.parse(pvd).findall(".//DataSet") for pvd in planes]
    end = min(max(float(frame.attrib["timestep"]) for frame in frames) for frames in collections)
    starts = []
    for pvd, frames in zip(planes, collections, strict=True):
        selected = [
            frame for frame in frames if end - period < float(frame.attrib["timestep"]) <= end
        ]
        starts.append(float(selected[0].attrib["timestep"]))
        grids = [pv.read(pvd.parent / frame.attrib["file"]) for frame in selected]
        mean = np.mean([np.asarray(grid["velocity"]) / speed for grid in grids], axis=0)
        records.append((grids[-1].points, mean))
    records.sort(key=lambda row: row[0][0] @ direction)
    axial = [mean @ direction for _, mean in records]
    vertical = [mean[:, 2] for _, mean in records]
    vertical_limit = max(np.max(np.abs(values)) for values in vertical)
    for row, values, limits, cmap, label in (
        (
            0,
            axial,
            (min(v.min() for v in axial), max(v.max() for v in axial)),
            "viridis",
            r"Mean $u_{\parallel}/U_\infty$",
        ),
        (1, vertical, (-vertical_limit, vertical_limit), "RdBu_r", r"Mean $u_z/U_\infty$"),
    ):
        for column, ((points, _), field) in enumerate(zip(records, values, strict=True)):
            artist = axes[row, column].tricontourf(
                points[:, 1],
                points[:, 2],
                field,
                levels=np.linspace(*limits, 25),
                cmap=cmap,
            )
            axes[row, column].set_aspect("equal")
        fig.colorbar(
            artist,
            ax=axes[row],
            label=label,
            format="%.2f",
            ticks=np.linspace(*limits, 5),
        )
    for column, (points, _) in enumerate(records):
        axes[0, column].set_title(f"x = {points[0, 0]:g} m")
        axes[0, column].set_ylabel("z [m]")
    for ax in axes[:, -1]:
        ax.set_xlabel("y [m]")
    fig.suptitle(f"Mean wake velocity\nt = {max(starts):.2f}–{end:.2f} s")
    _theme.save_fig(
        fig, figures_dir / "delta_wing_wake.png", figure_format=figure_format, bbox_inches=None
    )
