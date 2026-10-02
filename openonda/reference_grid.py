"""Reference-grid convergence calculations and thesis-sized force figures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def richardson_gci(grids: list[dict], metric: str) -> dict:
    invalid = {"valid": False, "order": None, "extrapolated": None, "fine_gci": None}
    if len(grids) < 3:
        return {**invalid, "reason": "fewer than three meshes"}
    coarse, medium, fine = grids[-3:]
    if not all(
        np.isfinite(grid[metric]) and np.isfinite(grid["h"]) and grid["h"] > 0
        for grid in (coarse, medium, fine)
    ):
        return {**invalid, "reason": "nonfinite metric or invalid spacing"}
    ratio_a = coarse["h"] / medium["h"]
    ratio_b = medium["h"] / fine["h"]
    coarse_difference = coarse[metric] - medium[metric]
    fine_difference = medium[metric] - fine[metric]
    if (
        ratio_a <= 1
        or ratio_b <= 1
        or not math.isclose(ratio_a, ratio_b, rel_tol=5.0e-3)
        or coarse_difference * fine_difference <= 0
        or fine_difference == 0
    ):
        return {**invalid, "reason": "mesh ratios or differences are not monotone"}

    order = math.log(abs(coarse_difference / fine_difference)) / math.log(ratio_b)
    if not np.isfinite(order) or order <= 0:
        return {**invalid, "reason": "differences do not decrease with refinement"}
    uncertainty = [
        grid.get("uncertainty_95", {}).get(metric) or 0.0 for grid in (coarse, medium, fine)
    ]
    if (
        abs(coarse_difference) <= uncertainty[0] + uncertainty[1]
        or abs(fine_difference) <= uncertainty[1] + uncertainty[2]
    ):
        return {**invalid, "reason": "spatial differences unresolved within sampling uncertainty"}
    denominator = ratio_b**order - 1.0
    extrapolated = fine[metric] + (fine[metric] - medium[metric]) / denominator
    fine_gci = (
        1.25 * abs((fine[metric] - medium[metric]) / max(abs(fine[metric]), 1.0e-14)) / denominator
    )
    return {"valid": True, "order": order, "extrapolated": extrapolated, "fine_gci": fine_gci}


def plot_force_grids(grids: list[dict], path: Path) -> None:
    """Save mean/shedding metrics and the two remaining force RMS measures."""
    import matplotlib.pyplot as plt

    from openonda.plotting import (
        COLORS,
        centered_subplots_adjust,
        export_figure,
        figure_size,
        fit_thesis_y_label_margins,
        set_thesis_style,
        validate_thesis_figure,
    )

    set_thesis_style()
    labels = {
        "mean_drag": r"$\overline{C_D}$",
        "rms_drag": r"$C_{D,\mathrm{rms}}$",
        "rms_lift": r"$C_{L,\mathrm{rms}}$",
        "rms_side": r"$C_{S,\mathrm{rms}}$",
        "strouhal": r"$St$",
    }
    for output, metrics, size in (
        (path, ("mean_drag", "rms_lift", "strouhal"), "stacked"),
        (
            path.with_name(path.stem + "_fluctuations" + path.suffix),
            ("rms_drag", "rms_side"),
            "wide_short",
        ),
    ):
        figure, axes = plt.subplots(len(metrics), 1, figsize=figure_size(size), sharex=True)
        if not all(grid.get("qualified_statistics", True) for grid in grids):
            print("Force statistics unqualified: see the grid qualification report.")
        for axis, metric in zip(axes, metrics, strict=True):
            axis.plot(
                [grid["h"] for grid in grids],
                [grid[metric] for grid in grids],
                "o-",
                color=COLORS["fvm"],
            )
            axis.set_ylabel(labels[metric])
            axis.grid(False)
        axes[-1].invert_xaxis()
        axes[-1].set_xlabel("h / D")
        centered_subplots_adjust(figure, outer=0.18, bottom=0.16, top=0.92, hspace=0.26)
        fit_thesis_y_label_margins(figure, axes)
        validate_thesis_figure(figure, axes)
        export_figure(figure, output)
        plt.close(figure)
