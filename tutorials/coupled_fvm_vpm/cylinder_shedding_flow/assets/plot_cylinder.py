#!/usr/bin/env python3
"""Plot the coupled-cylinder drag and lift histories."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openonda-matplotlib-cache")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    path = CASE_DIR / "samples" / "forces_history.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    time = np.asarray([float(row["time"]) for row in rows])
    drag = np.asarray([float(row["drag_coefficient"]) for row in rows])
    lift = np.asarray([float(row["lift_coefficient"]) for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot(time, drag, color="#1769aa", linewidth=1.2)
    axes[0].set_ylabel(r"$C_D$")
    axes[1].plot(time, lift, color="#c62828", linewidth=1.2)
    axes[1].set_ylabel(r"$C_L$")
    axes[1].set_xlabel(r"$tU_\infty/D$")
    for axis in axes:
        axis.grid(alpha=0.25)

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"cylinder_forces.{suffix}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
