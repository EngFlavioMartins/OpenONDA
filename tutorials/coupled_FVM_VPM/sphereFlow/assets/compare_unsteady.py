#!/usr/bin/env python3
"""Compare the hybrid sphere against its fully meshed reference.

An unsteady periodic flow has three quantities that converge and can therefore
carry a 1% target: mean drag, lift amplitude, and Strouhal number.  Comparing
instantaneous fields point-by-point does not -- two runs of the same periodic
flow drift in phase, and the pointwise difference then measures the phase, not
the coupling.  This script reports the converged metrics, states the phase lag
explicitly, and only then shows the phase-aligned profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _geometry as G  # noqa: E402

CASE = Path(__file__).resolve().parent.parent
HYBRID = CASE / "samples"
REFERENCE = CASE / "referenceFlow" / "samples"
FIGURES = CASE / "figures"
DPI = 300


def _forces(directory: Path) -> pd.DataFrame:
    for name in ("ibm_forces_history.csv", "forces_history.csv"):
        path = directory / name
        if path.exists():
            return pd.read_csv(path)
    raise SystemExit(f"No force history in {directory}")


def _window(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["time"] >= G.SHEDDING_START]


def strouhal(time: np.ndarray, signal: np.ndarray) -> float:
    """Dominant frequency of the lift signal, in D/U units."""
    signal = signal - signal.mean()
    if len(signal) < 16:
        return float("nan")
    dt = float(np.mean(np.diff(time)))
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freq = np.fft.rfftfreq(len(signal), dt)
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    if 0 < peak < len(freq) - 1:
        # Parabolic interpolation: the FFT bin spacing alone is far coarser
        # than the 1% target on St.
        a, b, c = spectrum[peak - 1 : peak + 2]
        offset = 0.5 * (a - c) / (a - 2.0 * b + c)
        return float((peak + offset) * (freq[1] - freq[0]))
    return float(freq[peak])


def best_lag(time: np.ndarray, left: np.ndarray, right: np.ndarray, span: float) -> float:
    lags = np.linspace(-span, span, 201)
    scores = []
    for lag in lags:
        shifted = np.interp(time - lag, time, right)
        a, b = left - left.mean(), shifted - shifted.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float(a @ b / denominator) if denominator > 0 else -1.0)
    return float(lags[int(np.argmax(scores))]), float(max(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)

    hybrid, reference = _forces(HYBRID), _forces(REFERENCE)
    hw, rw = _window(hybrid), _window(reference)
    if hw.empty or rw.empty:
        raise SystemExit(
            f"No samples after tU/D = {G.SHEDDING_START}; run both cases to completion."
        )

    lift = "Cl" if "Cl" in hw else "Cy"
    metrics = {}
    for name, frame in (("hybrid", hw), ("reference", rw)):
        time = frame["time"].to_numpy()
        metrics[name] = {
            "Cd_mean": float(frame["Cd"].mean()),
            "Cd_rms": float(frame["Cd"].std()),
            "Cl_mean": float(frame[lift].mean()),
            "Cl_rms": float(np.sqrt((frame[lift] ** 2).mean())),
            "St": strouhal(time, frame[lift].to_numpy()),
            "samples": int(len(frame)),
        }

    grid = np.linspace(
        max(hw["time"].min(), rw["time"].min()), min(hw["time"].max(), rw["time"].max()), 2048
    )
    lag, correlation = best_lag(
        grid,
        np.interp(grid, hw["time"], hw[lift]),
        np.interp(grid, rw["time"], rw[lift]),
        span=2.0,
    )

    def relative(key):
        a, b = metrics["hybrid"][key], metrics["reference"][key]
        return 100.0 * (a - b) / b if b else float("nan")

    summary = {
        "window_start": G.SHEDDING_START,
        "hybrid": metrics["hybrid"],
        "reference": metrics["reference"],
        "relative_error_percent": {k: relative(k) for k in ("Cd_mean", "Cl_rms", "St")},
        "lift_phase_lag": lag,
        "lift_correlation_at_best_lag": correlation,
        "literature": G.LITERATURE,
    }
    (HYBRID / "comparison_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nConverged metrics over tU/D >= {G.SHEDDING_START}")
    print(f"{'':12}{'hybrid':>12}{'reference':>12}{'error %':>10}{'literature':>12}")
    for key, lit in (("Cd_mean", "Cd"), ("Cl_rms", None), ("St", "St")):
        print(
            f"  {key:10}{metrics['hybrid'][key]:12.4f}{metrics['reference'][key]:12.4f}"
            f"{relative(key):10.2f}"
            f"{G.LITERATURE.get(lit, float('nan')) if lit else float('nan'):12.4f}"
        )
    print(f"  lift phase lag {lag:+.3f} D/U at r = {correlation:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), dpi=DPI, sharex=True)
    for name, frame, colour in (("hybrid", hybrid, "C0"), ("reference", reference, "C1")):
        axes[0].plot(frame["time"], frame["Cd"], colour, lw=1.0, label=name)
        axes[1].plot(frame["time"], frame[lift], colour, lw=1.0, label=name)
    for axis in axes:
        axis.axvline(G.SHEDDING_START, color="0.6", ls=":", lw=0.8)
    axes[0].axhline(G.LITERATURE["Cd"], color="0.4", ls="--", lw=0.8, label="Johnson & Patel 1999")
    axes[0].set(ylabel="$C_d$", title="Sphere Re = 300, hybrid vs fully meshed")
    axes[1].set(xlabel="$tU/D$", ylabel=f"${lift}$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"forces.{args.format}")
    plt.close(fig)
    print(f"\nWrote {FIGURES / f'forces.{args.format}'} and samples/comparison_metrics.json")


if __name__ == "__main__":
    main()
