"""Plot NACA 4412 force coefficients in freestream-aligned axes."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openonda import plotting as theme

CASE_DIR = Path(__file__).resolve().parents[1]


def _wind_direction(case_dir: Path) -> np.ndarray:
    metadata_path = case_dir / "solution" / "run_metadata.json"
    if metadata_path.is_file():
        velocity = json.loads(metadata_path.read_text())["physics"]["freestream_velocity"]
    else:
        velocity = runpy.run_path(str(case_dir / "setup.py"))["FREESTREAM_VELOCITY"]
    vector = np.asarray(velocity, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("freestream velocity must be a finite three-component vector")
    speed = float(np.linalg.norm(vector[:2]))
    if speed <= 0.0 or not np.isclose(vector[2], 0.0):
        raise ValueError("freestream velocity must lie in the nonzero airfoil plane")
    return vector[:2] / speed


def _wind_axis_coefficients(
    body_drag: np.ndarray, body_lift: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    drag = body_drag * direction[0] + body_lift * direction[1]
    lift = -body_drag * direction[1] + body_lift * direction[0]
    return drag, lift


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
    drag, lift = _wind_axis_coefficients(
        body_axis_drag_coefficient, body_axis_lift_coefficient, _wind_direction(CASE_DIR)
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

    last_half = time >= 0.5 * time[-1]
    print(
        f"Mean over last half of saved time window: wind-axis drag_coefficient={np.mean(drag[last_half]):.4f}; "
        f"lift_coefficient={np.mean(lift[last_half]):.4f}."
    )


if __name__ == "__main__":
    main()
