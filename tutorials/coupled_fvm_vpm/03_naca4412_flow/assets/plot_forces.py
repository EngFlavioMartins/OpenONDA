"""Plot NACA 4412 force coefficients in freestream-aligned axes."""

from __future__ import annotations

import argparse
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as theme
from openonda.results import planar_direction, read_csv_columns, read_json

CASE_DIR = Path(__file__).resolve().parents[1]


def _wind_direction(case_dir):
    velocity = read_json(case_dir / "solution/run_metadata.json")["physics"]["freestream_velocity"]
    return planar_direction(velocity, axes=(0, 1))


def _wind_axis_coefficients(
    body_drag: np.ndarray, body_lift: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    drag = body_drag * direction[0] + body_lift * direction[1]
    lift = -body_drag * direction[1] + body_lift * direction[0]
    return drag, lift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=theme.FORMAT_CHOICES, default="both")
    args = parser.parse_args()

    theme.set_thesis_style()
    source = CASE_DIR / "samples" / "ibm_forces_history.csv"
    data = read_csv_columns(source)
    time = data["time"]
    body_axis_drag_coefficient = data["drag_coefficient"]
    body_axis_lift_coefficient = data["lift_coefficient"]
    slip_error = data["slip_error"]
    drag, lift = _wind_axis_coefficients(
        body_axis_drag_coefficient, body_axis_lift_coefficient, _wind_direction(CASE_DIR)
    )

    figures = CASE_DIR / "figures"
    figure, axes = plt.subplots(2, 1, figsize=theme.figure_size("stacked"), sharex=True)
    axes[0].plot(time, drag, label=r"$C_D$")
    axes[0].plot(time, lift, label=r"$C_L$")
    axes[0].set_ylabel("wind-axis coefficient")
    axes[0].legend()
    axes[0].grid(False)
    axes[1].semilogy(time, np.maximum(slip_error, 1e-16))
    axes[1].set(xlabel="time", ylabel="IBM no-slip error")
    axes[1].grid(False)
    theme.centered_subplots_adjust(figure, outer=0.20, bottom=0.14, top=0.94, hspace=0.32)
    output = figures / f"force_history.{args.format}"
    theme.export_figure(figure, output, figure_format=args.format)
    plt.close(figure)

    last_half = time >= 0.5 * time[-1]
    print(
        f"Mean over last half of saved time window: wind-axis drag_coefficient={np.mean(drag[last_half]):.4f}; "
        f"lift_coefficient={np.mean(lift[last_half]):.4f}."
    )


if __name__ == "__main__":
    main()
