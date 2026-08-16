#!/usr/bin/env python3
"""Plot cylinder coupling cost, particle population, and transfer quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    return np.asarray([row.get(section, {}).get(key, 0.0) for row in records], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    path = util.SOLUTION / "coupler_diagnostics.jsonl"
    if not path.exists():
        raise SystemExit("No coupling diagnostics found.")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    time = np.asarray([row["time"] for row in records])

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6), sharex=True, constrained_layout=True)
    vpm = _values(records, "timing_seconds", "vpm")
    fvm = _values(records, "timing_seconds", "fvm")
    transfer = sum(
        (_values(records, "timing_seconds", key) for key in ("donor", "fringe", "handoff")),
        start=np.zeros_like(time),
    )
    axes[0, 0].stackplot(time, vpm, fvm, transfer, labels=("VPM", "FVM", "transfer"), alpha=0.85)
    axes[0, 0].set(ylabel="wall time / step [s]", title="Coupling cost")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(time, [row.get("handoff_particle_count", 0) for row in records], label="total")
    axes[0, 1].plot(time, _values(records, "handoff", "n_free"), ls="--", label="free wake")
    axes[0, 1].set(ylabel="particles", title="Particle population")
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].plot(time, _values(records, "handoff", "flux_ratio"))
    axes[1, 0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[1, 0].set(xlabel="flow time", ylabel="flux ratio", title="FVM-to-VPM vorticity flux")
    angular = np.asarray(
        [
            row.get("conservation", {}).get("corrected_mismatch", {}).get("angular_impulse", 0.0)
            for row in records
        ],
        dtype=float,
    )
    axes[1, 1].semilogy(time, np.maximum(angular, 1e-16))
    axes[1, 1].set(
        xlabel="flow time",
        ylabel="angular-impulse mismatch",
        title="Unconstrained 3-D handoff moment",
    )
    util.save(fig, "coupling_diagnostics", args.format)
    plt.close(fig)


if __name__ == "__main__":
    main()
