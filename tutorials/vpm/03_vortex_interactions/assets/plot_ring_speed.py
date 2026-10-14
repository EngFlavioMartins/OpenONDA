"""Single-ring translation against Saffman's Gaussian-core speed.

U = Gamma / (4 pi R) [ln(8 R / a) - 0.558],  a^2 = a0^2 + 4 nu t,
for omega = Gamma / (pi a^2) exp(-r^2 / a^2) (Saffman 1970, a << R).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import setup
from assets.postprocess import figure_size, read_csv_table, save_figure, theme, write_text

FIGURE_NAME = "ring_speed"


def saffman_speed(time, ring_radius):
    core = np.sqrt(setup.CORE_RADIUS**2 + 4.0 * setup.KINEMATIC_VISCOSITY * time)
    return (
        setup.RING_CIRCULATION
        / (4.0 * np.pi * ring_radius)
        * (np.log(8.0 * ring_radius / core) - 0.558)
    )


def plot(output, auxiliary_output, formats):
    plotting = theme()
    plotting.set_thesis_style()
    frame = pd.DataFrame(
        read_csv_table(setup.TUTORIAL_DIR / "samples/single_ring/ring_diagnostics.csv")
    )
    frame = frame.sort_values("time")
    time = frame.time.to_numpy()
    position = frame.vortex_centroid_x.to_numpy()
    radius = frame.impulse_radius.to_numpy()
    # The start-up transient of the particle cores settles within 0.2 s.
    window = time >= 0.2
    speed = np.polyfit(time[window], position[window], 1)[0]
    reference = saffman_speed(time, radius)
    reference_position = position[0] + np.concatenate(
        ([0.0], np.cumsum(0.5 * (reference[1:] + reference[:-1]) * np.diff(time)))
    )
    reference_speed = float(np.mean(reference[window]))
    error = speed / reference_speed - 1.0

    fig, ax = plt.subplots(figsize=figure_size(6.0))
    ax.plot(time, position, color=plotting.COLORS["vpm"], lw=1.2, label="VPM")
    ax.plot(
        time,
        reference_position,
        color=plotting.COLORS["reference"],
        lw=1.0,
        ls="--",
        label="Saffman (1970)",
    )
    ax.set(xlabel="$t$ [s]", ylabel="$\\bar{x}$ [m]")
    ax.legend(frameon=False)
    ax.text(0.04, 0.94, f"$U/U_S-1={100 * error:+.1f}\\,\\%$", transform=ax.transAxes, va="top")
    plotting.centered_subplots_adjust(fig, outer=0.16, bottom=0.2, top=0.95)
    save_figure(fig, output / FIGURE_NAME, ax, formats)
    plt.close(fig)
    summary = {
        "figure": FIGURE_NAME,
        "speed": float(speed),
        "saffman_speed": reference_speed,
        "relative_error": float(error),
        "fit_window_s": [0.2, float(time[-1])],
    }
    write_text(auxiliary_output / f"{FIGURE_NAME}.json", json.dumps(summary, indent=2) + "\n")
    print(f"ring speed {speed:.4f} m/s, Saffman {reference_speed:.4f} m/s ({100 * error:+.2f} %)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=setup.TUTORIAL_DIR / "figures")
    parser.add_argument(
        "--auxiliary-output", type=Path, default=setup.TUTORIAL_DIR / "figures/auxiliary"
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    args = parser.parse_args()
    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    plot(args.output, args.auxiliary_output, formats)


if __name__ == "__main__":
    main()
