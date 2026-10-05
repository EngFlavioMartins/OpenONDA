#!/usr/bin/env python3
"""Compare reference, coupled-FVM, and VPM profiles at every common saved time."""

import argparse


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.integrate import trapezoid

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    centered_subplots_adjust,
    requested_formats,
    set_thesis_style,
)

from . import postprocess as data
from .velocity_profile_data import coincident_velocity_profiles, profile_geometry


def available_transverse_profiles(reference_directory):
    """Use saved profile definitions, including optional near-body FVM profiles."""
    result = []
    for reference in reference_directory.glob("transverse_x*.csv"):
        x_position = float(reference.stem.removeprefix("transverse_x"))
        vpm = data.CASE_DIR / "samples" / f"vpm_{reference.name}"
        fvm = data.CASE_DIR / "samples" / f"fvm_{reference.name}"
        result.append((x_position, reference, vpm, fvm))
    result.sort(key=lambda row: row[0])
    return result


def plot_frame(profiles, samples, name, figure_format, geometry):
    figure, axes = plt.subplots(
        2,
        len(profiles),
        figsize=(12.5 * CM, 12.5 * CM),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    errors: dict[str, dict[str, float]] = {}
    for column, (x_position, reference_path, vpm_path, fvm_path) in enumerate(profiles):
        reference = samples[reference_path]
        vpm = samples[vpm_path]
        candidates = [("Coupled VPM", vpm, COLORS["vpm"], "o")]
        if fvm_path is not None:
            candidates.insert(0, ("Coupled FVM", samples[fvm_path], COLORS["fvm"], "s"))
        for row, velocity in enumerate(data.VELOCITY_COLUMNS[:2]):
            axis = axes[row, column]
            for box, color, alpha in (
                (geometry["fvm_box"], COLORS["background_light"], 0.3),
                (geometry["transfer_box"], COLORS["background_strong"], 0.25),
            ):
                if box["xmin"] <= x_position <= box["xmax"]:
                    axis.axvspan(
                        box["ymin"] / geometry["diameter"],
                        box["ymax"] / geometry["diameter"],
                        color=color,
                        alpha=alpha,
                        zorder=0,
                    )
            axis.plot(
                reference.position_y / geometry["diameter"],
                reference[velocity] / geometry["speed"],
                color=COLORS["reference"],
                linewidth=REFERENCE_LINE_WIDTH,
                label="Reference flow",
                linestyle="--",
            )
            for label, candidate, color, marker in candidates:
                axis.plot(
                    candidate.position_y / geometry["diameter"],
                    candidate[velocity] / geometry["speed"],
                    color=color,
                    linewidth=LINE_WIDTH,
                    label=label,
                    marker=marker,
                    markevery=6,
                    markersize=2.5,
                )
                lower = max(float(reference.position_y.min()), float(candidate.position_y.min()))
                upper = min(float(reference.position_y.max()), float(candidate.position_y.max()))
                y = candidate.position_y.to_numpy(dtype=float)
                keep = (y >= lower) & (y <= upper)
                y = y[keep]
                actual = candidate[velocity].to_numpy(dtype=float)[keep]
                expected = np.interp(y, reference.position_y, reference[velocity])
                difference = (actual - expected) / geometry["speed"]
                errors[f"{label.lower().replace(' ', '_')}_x{x_position:g}_{velocity}"] = {
                    "rms": float(np.sqrt(trapezoid(difference**2, y) / (y[-1] - y[0]))),
                    "maximum": float(np.abs(difference).max()),
                }
            axis.grid(False)
            axis.xaxis.set_major_locator(MaxNLocator(5))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            if row == 1:
                axis.set_xlabel(rf"$y/D,\quad x/D={x_position / geometry['diameter']:g}$")
        axes[0, 0].set_ylabel(r"$u/U_\infty$")
        axes[1, 0].set_ylabel(r"$v/U_\infty$")

    legend = {}
    for axis in axes[0]:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    figure.legend(
        legend.values(),
        legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.12),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )
    figure.legend(
        [
            Patch(color=COLORS["background_light"], alpha=0.3),
            Patch(color=COLORS["background_strong"], alpha=0.25),
        ],
        [
            rf"FVM domain ($x_{{\max}}/D={geometry['fvm_box']['xmax'] / geometry['diameter']:g}$)",
            "Transfer region",
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )
    centered_subplots_adjust(
        figure,
        outer=0.1205,
        bottom=0.28,
        top=0.93,
        hspace=0.18,
        wspace=0.16,
    )

    data.save_figure(figure, axes.flat, name, figure_format)
    plt.close(figure)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    arguments = parser.parse_args()

    reference_directory = data.reference_directory()
    profiles = available_transverse_profiles(reference_directory)
    geometry = profile_geometry()
    formats = requested_formats(arguments.format)
    frames = []
    set_thesis_style()
    for time, frame_profiles, samples, source_information in coincident_velocity_profiles(
        profiles, geometry
    ):
        name = f"velocity_profiles_t{time:.12g}"
        errors = plot_frame(frame_profiles, samples, name, arguments.format, geometry)
        frames.append(
            {
                "time": time,
                "files": [f"{name}.{ext}" for ext in formats],
                "errors": errors,
                "native_fvm_profiles": source_information,
            }
        )

    data.write_json(
        "velocity_profile_errors.json",
        {
            "reference": str(reference_directory.relative_to(data.CASE_DIR)),
            "time_alignment": "common saved physical states; clock roundoff only, no time interpolation",
            "profile_positions_x": [row[0] for row in profiles],
            "normalization": {
                "diameter": geometry["diameter"],
                "freestream_speed": geometry["speed"],
            },
            "coupled_fvm_domain": geometry["fvm_box"],
            "transfer_region": geometry["transfer_box"],
            "frames": frames,
        },
    )
    print(
        f"Profile comparison: {len(frames)} common saved states, "
        f"t={frames[0]['time']:g}–{frames[-1]['time']:g} s."
    )


if __name__ == "__main__":
    main()
