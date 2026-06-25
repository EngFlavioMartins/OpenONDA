#!/usr/bin/env python3
"""Leapfrogging-ring trajectory comparison — ``compare_trajectory.png``.

Overlays the circulation-weighted ring trajectory (R/R₀ vs x/R₀) of every
leapfrogging case found under ``solution/`` against the LBM literature
reference.  Each case keeps one colour + linestyle for *both* of its rings
(see :func:`_common.case_style`); the LBM reference is drawn as a grey dotted
line so it never collides with a simulation style.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
sys.path.insert(0, str(ASSETS_DIR))
from _common import CM, case_style, discover_cases, load_theme, save_fig  # noqa: E402

REFERENCE = ASSETS_DIR / "references" / "leapfrogging_lbm_trajectory.csv"
C_REF = "#505050"


def _step(path: Path) -> int:
    match = re.search(r"_(\d+)\.h5$", path.name)
    return int(match.group(1)) if match else -1


def load_trajectory(case_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Circulation-weighted axial position and radius for each ring over time."""
    rows: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for path in sorted(case_dir.glob("vpm_*_*.h5"), key=_step):
        with h5py.File(path, "r") as handle:
            particles = handle["particles"]
            position = particles["position"][:]
            circulation = particles["circulation"][:]
            group = particles["group_id"][:]
        weight = np.linalg.norm(circulation, axis=1)
        for ring in rows:
            mask = group == ring
            if not np.any(mask) or np.sum(weight[mask]) <= 0.0:
                continue
            axial = np.average(position[mask, 0], weights=weight[mask])
            radius = np.hypot(position[mask, 1], position[mask, 2])
            rows[ring].append((float(axial), float(np.average(radius, weights=weight[mask]))))
    return {
        ring: (np.asarray(values)[:, 0], np.asarray(values)[:, 1])
        for ring, values in rows.items()
        if values
    }


def plot_reference(ax) -> bool:
    """Draw both LBM reference rings as one grey dotted source. Returns success."""
    if not REFERENCE.exists():
        print(f"  [WARNING] LBM reference not found: {REFERENCE}")
        return False
    reference = pd.read_csv(REFERENCE)
    for ring in sorted(reference["ring"].unique()):
        data = reference[reference["ring"] == ring]
        ax.plot(data["x_over_R0"], data["R_over_R0"], color=C_REF, linestyle=":", lw=1.0)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-dir", default=str(CASE_DIR / "solution"))
    parser.add_argument("--figures-dir", default=str(CASE_DIR / "figures"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    load_theme()
    fig, ax = plt.subplots(figsize=(12.8 * CM, 7.5 * CM))

    has_ref = plot_reference(ax)

    legend_handles: list[Line2D] = []
    for case_dir in discover_cases(args.solution_dir, family="leapfrog"):
        trajectory = load_trajectory(case_dir)
        if not trajectory:
            continue
        st = case_style(case_dir.name)
        for axial, radius in trajectory.values():
            ax.plot(
                axial, radius,
                color=st["color"], linestyle=st["linestyle"],
                lw=1.1, marker=st["marker"], ms=3, markevery=5, mew=0.4,
            )
        legend_handles.append(
            Line2D([0], [0], color=st["color"], linestyle=st["linestyle"],
                   marker=st["marker"], ms=4, lw=1.1, label=st["label"])
        )

    if has_ref:
        legend_handles.append(
            Line2D([0], [0], color=C_REF, linestyle=":", lw=1.0, label="LBM reference")
        )

    ax.set_xlabel(r"Axial position, $x/R_0$")
    ax.set_ylabel(r"Ring radius, $R/R_0$")
    ax.set_ylim([0.5,1.5])
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=10, ncol=2, loc="lower right")

    save_fig(fig, Path(args.figures_dir) / "rings_trajectory.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
