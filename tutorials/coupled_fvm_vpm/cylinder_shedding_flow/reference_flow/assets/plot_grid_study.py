#!/usr/bin/env python3
"""Plot the cylinder grid-study force statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openonda-matplotlib-cache")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    report = json.loads((CASE_DIR / "solution" / "grid_study.json").read_text(encoding="utf-8"))
    records = report["cases"]
    dx = np.asarray([record["dx"] for record in records])
    mean_cd = np.asarray([record["mean_cd"] for record in records])
    cl_rms = np.asarray([record["cl_rms"] for record in records])
    strouhal = np.asarray([record["strouhal"] for record in records])

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for axis, values, label in zip(
        axes,
        (mean_cd, cl_rms, strouhal),
        (r"$\overline{C_D}$", r"$C_{L,\mathrm{rms}}$", r"$St$"),
        strict=True,
    ):
        axis.plot(dx, values, "o-", color="#1769aa", linewidth=1.5)
        axis.set_xlabel(r"wall spacing $\Delta x/D$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.invert_xaxis()

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"grid_study.{suffix}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
