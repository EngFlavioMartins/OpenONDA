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
    production_names = set(report["production_cases"])
    production = [record for record in records if record["case"] in production_names]
    dx = np.asarray([record["dx"] for record in records])
    production_dx = np.asarray([record["dx"] for record in production])

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    for axis, metric, label in zip(
        axes,
        ("mean_cd", "cl_rms", "strouhal"),
        (r"$\overline{C_D}$", r"$C_{L,\mathrm{rms}}$", r"$St$"),
        strict=True,
    ):
        values = np.asarray([record[metric] for record in records])
        production_values = np.asarray([record[metric] for record in production])
        axis.plot(dx, values, "o--", color="#9aa4b2", linewidth=1.0, label="all-grid trend")
        axis.plot(
            production_dx,
            production_values,
            "o-",
            color="#1769aa",
            linewidth=1.8,
            label="r=1.5 production grids",
        )
        convergence = report["grid_convergence"][metric]
        extrapolated = convergence["richardson_extrapolated_value"]
        if extrapolated is not None:
            axis.axhline(
                extrapolated,
                color="#d1495b",
                linewidth=1.0,
                linestyle=":",
                label="Richardson limit",
            )
        gci = convergence["fine_grid_gci_percent"]
        status = (
            f"GCI$_f$={gci:.2f}%"
            if gci is not None
            else convergence["status"].replace("_", " ").capitalize()
        )
        axis.set_title(status, fontsize=9)
        axis.set_xlabel(r"wall spacing $\Delta x/D$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    axes[0].legend(fontsize=7, loc="best")

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"grid_study.{suffix}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
