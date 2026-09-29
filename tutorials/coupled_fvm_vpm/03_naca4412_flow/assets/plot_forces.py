"""Plot NACA 4412 force coefficients in freestream-aligned axes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as theme

CASE_DIR = Path(__file__).resolve().parents[1]
ALPHA = math.radians(10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=theme.EXPORT_FORMATS, default="png")
    args = parser.parse_args()

    theme.set_thesis_style()
    source = CASE_DIR / "samples" / "ibm_forces_history.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No force samples found in {source}")
    data = np.array(
        [
            [
                float(row[key])
                for key in (
                    "time",
                    "drag_coefficient",
                    "lift_coefficient",
                    "slip_error",
                )
            ]
            for row in rows
        ]
    )
    time, body_axis_drag_coefficient, body_axis_lift_coefficient, slip_error = data.T
    drag = body_axis_drag_coefficient * math.cos(ALPHA) + body_axis_lift_coefficient * math.sin(
        ALPHA
    )
    lift = -body_axis_drag_coefficient * math.sin(ALPHA) + body_axis_lift_coefficient * math.cos(
        ALPHA
    )

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), sharex=True)
    axes[0].plot(time, drag, label=r"$C_D$")
    axes[0].plot(time, lift, label=r"$C_L$")
    axes[0].set_ylabel("wind-axis coefficient")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(time, np.maximum(slip_error, 1e-16))
    axes[1].set(xlabel="time", ylabel="IBM no-slip error")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output = figures / f"force_history.{args.format}"
    theme.fit_thesis_y_label_margins(figure, axes)
    theme.validate_thesis_figure(figure, axes)
    figure.savefig(output, dpi=theme.DEFAULT_DPI, bbox_inches=None)
    plt.close(figure)
    print(f"Wrote {output}")

    settled = time >= 0.5 * time[-1]
    print(
        f"Settled wind-axis mean drag_coefficient={np.mean(drag[settled]):.4f}; "
        f"lift_coefficient={np.mean(lift[settled]):.4f}."
    )


if __name__ == "__main__":
    main()
