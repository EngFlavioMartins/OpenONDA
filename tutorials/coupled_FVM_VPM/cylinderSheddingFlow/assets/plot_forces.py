"""Plot cylinder force coefficients and report the resolved shedding frequency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()

    source = CASE_DIR / "samples" / "ibm_forces_history.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No force samples found in {source}")
    data = np.array([[float(row[key]) for key in ("time", "Cd", "Cl", "slip")] for row in rows])
    time, cd, cl, slip = data.T

    figures = CASE_DIR / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(time, cd, label=r"$C_D$")
    axes[0].plot(time, cl, label=r"$C_L$")
    axes[0].set_ylabel("force coefficient")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(time, np.maximum(slip, 1e-16))
    axes[1].set(xlabel="time", ylabel="IBM no-slip error")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output = figures / f"force_history.{args.format}"
    figure.savefig(output, dpi=180)
    print(f"Wrote {output}")

    settled = time >= 0.5 * time[-1]
    if np.count_nonzero(settled) >= 16:
        sample_time = time[settled]
        signal = cl[settled] - np.mean(cl[settled])
        frequencies = np.fft.rfftfreq(len(signal), d=float(np.median(np.diff(sample_time))))
        spectrum = np.abs(np.fft.rfft(signal))
        peak = 1 + int(np.argmax(spectrum[1:]))
        print(
            f"Settled mean Cd={np.mean(cd[settled]):.4f}; "
            f"dominant Strouhal St=fD/U={frequencies[peak]:.4f}."
        )


if __name__ == "__main__":
    main()
