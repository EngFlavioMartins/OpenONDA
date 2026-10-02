#!/usr/bin/env python3
"""Compare reference, coupled-FVM, and VPM profiles at one common saved time."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    centered_subplots_adjust,
    set_thesis_style,
)

from . import postprocess as data


def available_transverse_profiles(reference_directory):
    """Use saved profile definitions, including optional near-body FVM profiles."""
    result = []
    for reference in reference_directory.glob("transverse_x*.csv"):
        match = re.fullmatch(r"transverse_x([-+0-9.eE]+)\.csv", reference.name)
        if match is None:
            continue
        x_position = float(match.group(1))
        if not np.isfinite(x_position):
            raise ValueError(f"Non-finite transverse profile coordinate: {reference}")
        vpm = data.CASE_DIR / "samples" / f"vpm_{reference.name}"
        if not vpm.is_file():
            continue
        fvm = data.CASE_DIR / "samples" / f"fvm_{reference.name}"
        if not fvm.is_file():
            fvm = data.CASE_DIR / "samples" / reference.name
        result.append((x_position, reference, vpm, fvm if fvm.is_file() else None))
    result.sort(key=lambda row: row[0])
    if not result:
        raise ValueError("No matching saved reference/VPM transverse profiles are available")
    if len({row[0] for row in result}) != len(result):
        raise ValueError("Ambiguous transverse profile filenames identify the same x coordinate")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    arguments = parser.parse_args()

    reference_directory = data.reference_directory()
    profiles = available_transverse_profiles(reference_directory)
    time = data.latest_common_profile_time(
        tuple(path for _, *paths in profiles for path in paths if path is not None)
    )

    set_thesis_style()
    figure, axes = plt.subplots(
        2,
        len(profiles),
        figsize=(12.5 * CM, 12.0 * CM),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    errors: dict[str, dict[str, float]] = {}
    for column, (x_position, reference_path, vpm_path, fvm_path) in enumerate(profiles):
        reference = data.profile(reference_path, time)
        vpm = data.profile(vpm_path, time)
        candidates = [("VPM", vpm, COLORS["vpm"])]
        if fvm_path is not None:
            candidates.insert(0, ("Coupled FVM", data.profile(fvm_path, time), COLORS["fvm"]))
        for row, velocity in enumerate(data.VELOCITY_COLUMNS[:2]):
            axis = axes[row, column]
            axis.plot(
                reference.position_y,
                reference[velocity],
                color=COLORS["reference"],
                linewidth=REFERENCE_LINE_WIDTH,
                label="Reference FVM",
                linestyle="--",
            )
            for label, candidate, color in candidates:
                axis.plot(
                    candidate.position_y,
                    candidate[velocity],
                    color=color,
                    linewidth=LINE_WIDTH,
                    label=label,
                )
                lower = max(float(reference.position_y.min()), float(candidate.position_y.min()))
                upper = min(float(reference.position_y.max()), float(candidate.position_y.max()))
                y = candidate.position_y.to_numpy(dtype=float)
                keep = (y >= lower) & (y <= upper)
                y = y[keep]
                if len(y) < 2 or y[-1] <= y[0]:
                    raise ValueError(
                        f"Profiles have insufficient common spatial support at x/D={x_position:g}"
                    )
                actual = candidate[velocity].to_numpy(dtype=float)[keep]
                expected = np.interp(y, reference.position_y, reference[velocity])
                difference = actual - expected
                errors[f"{label.lower().replace(' ', '_')}_x{x_position:g}_{velocity}"] = {
                    "rms": float(np.sqrt(trapezoid(difference**2, y) / (y[-1] - y[0]))),
                    "maximum": float(np.abs(difference).max()),
                }
            axis.grid(False)
            if row == 1:
                axis.set_xlabel(rf"$y/D,\quad x/D={x_position:g}$")
        axes[0, 0].set_ylabel(r"$u/U_\infty$")
        axes[1, 0].set_ylabel(r"$v/U_\infty$")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
    )
    data.write_json(
        "reference_profile_errors.json",
        {
            "reference": str(reference_directory.relative_to(data.CASE_DIR)),
            "time": time,
            "time_alignment": "common saved physical state; clock roundoff only, no time interpolation",
            "scope": "instantaneous profiles; not a time-averaged or phase-convergence claim",
            "profile_positions_x": [row[0] for row in profiles],
            "errors": errors,
        },
    )
    centered_subplots_adjust(
        figure,
        outer=0.1205,
        bottom=0.205,
        top=0.99,
        hspace=0.18,
        wspace=0.16,
    )

    data.save_figure(figure, axes.flat, "reference_profiles", arguments.format)
    print(f"Profile comparison: latest common saved state t={time:g} s.")


if __name__ == "__main__":
    main()
