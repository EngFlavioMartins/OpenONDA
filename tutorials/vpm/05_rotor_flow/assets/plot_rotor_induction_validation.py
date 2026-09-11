#!/usr/bin/env python3
"""Compare required 1D/2D induction with the finite-distance theory."""

from __future__ import annotations

if not __package__:
    from pathlib import Path
    from openonda.tutorial_runner import case_package

    __package__ = case_package(Path(__file__).resolve().parents[1]) + ".assets"

import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as theme
from ._common import FIGURES_DIR, build_arg_parser, rotor_inputs
from .finite_distance_theory import (
    AXIAL_REFERENCE_FLOOR,
    AXIAL_RELATIVE_TOLERANCE,
    TANGENTIAL_REFERENCE_FLOOR,
    TANGENTIAL_RELATIVE_TOLERANCE,
)
from .plot_rotor_wake_planes import finite_distance_profiles


def _error(actual, reference, floor):
    valid = np.isfinite(actual) & np.isfinite(reference) & (np.abs(reference) >= floor)
    if not np.any(valid):
        return np.nan
    scaled = (actual[valid] - reference[valid]) / np.maximum(np.abs(reference[valid]), floor)
    return float(np.sqrt(np.mean(scaled**2)))


def _rms(actual, reference):
    valid = np.isfinite(actual) & np.isfinite(reference)
    if not np.any(valid):
        return np.nan, 0, len(actual)
    return (
        float(np.sqrt(np.mean((actual[valid] - reference[valid]) ** 2))),
        int(valid.sum()),
        len(actual),
    )


def main():
    args = build_arg_parser(__doc__).parse_args()
    inputs = rotor_inputs()
    records = finite_distance_profiles(inputs)
    theme.set_thesis_style()
    fig, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"))
    colors = theme.COLORS
    for row in records:
        station_radius = getattr(inputs, "station_radius", inputs.rotor_radius)
        label = f"{row['x'] / (2 * station_radius):g}D"
        line_style = "-" if row["complete"] else ":"
        axes[0].plot(
            row["radius"],
            row["actual_axial_induction"],
            color=colors["VPMpurple"],
            ls=line_style,
            label=f"native {label}",
        )
        axes[0].plot(
            row["radius"],
            row["reference_axial_induction"],
            color=colors["reference"],
            ls="--",
            label=f"VC theory {label}",
        )
        axes[1].plot(
            row["radius"],
            row["actual_tangential_induction"],
            color=colors["VPMpurple"],
            ls=line_style,
            label=f"native {label}",
        )
        axes[1].plot(
            row["radius"],
            row["reference_tangential_induction"],
            color=colors["reference"],
            ls="--",
            label=f"VC theory {label}",
        )
        axial_rms, axial_count, axial_total = _rms(
            row["actual_axial_velocity"], row["reference_axial_velocity"]
        )
        azimuthal_rms, azimuthal_count, azimuthal_total = _rms(
            row["actual_tangential_velocity"], row["reference_tangential_velocity"]
        )
        print(
            f"{row['name']}: axial RMS={axial_rms:.4g} m/s ({axial_count}/{axial_total}), "
            f"scaled={_error(row['actual_axial_induction'], row['reference_axial_induction'], AXIAL_REFERENCE_FLOOR):.3f}; "
            f"azimuthal RMS={azimuthal_rms:.4g} m/s ({azimuthal_count}/{azimuthal_total}), "
            f"scaled={_error(row['actual_tangential_induction'], row['reference_tangential_induction'], TANGENTIAL_REFERENCE_FLOOR):.3f}"
        )
    axes[0].set(ylabel="Axial induction, $a$", xlim=(0, 2.0))
    axes[1].set(ylabel="Azimuthal induction, $a^\\prime$", xlabel="$r/R$, radius", xlim=(0, 2.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        fontsize="small",
    )
    fig.suptitle(
        "Finite-distance right-vortex-cylinder reference; "
        f"diagnostic screens {AXIAL_RELATIVE_TOLERANCE:.0%}/{TANGENTIAL_RELATIVE_TOLERANCE:.0%}; not qualification gates",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 0.96))
    theme.save_fig(
        fig,
        FIGURES_DIR / "rotor_induction_validation.png",
        figure_format=args.format,
        dpi=args.dpi,
        bbox_inches=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
