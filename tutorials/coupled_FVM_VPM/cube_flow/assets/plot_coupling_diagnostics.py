#!/usr/bin/env python3
"""Plot coupled-run timing, particle population, and transfer diagnostics."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_DPI = 300


def _records() -> list[dict]:
    path = util.SOLUTION / "coupler_diagnostics.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A live writer may leave one temporarily incomplete final line.
                continue
    return records


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    # Unevaluated diagnostics are null, not zero; NaN keeps them off the plot.
    return np.asarray(
        [
            np.nan
            if row.get(section, {}).get(key, 0.0) is None
            else row.get(section, {}).get(key, 0.0)
            for row in records
        ],
        dtype=float,
    )


def _evaluated(records: list[dict]) -> np.ndarray:
    """Steps on which the on-cadence transfer diagnostics were actually computed."""
    return np.asarray(
        [bool(row.get("transfer", {}).get("diagnostics_evaluated", True)) for row in records]
    )


def plot(figure_format: str) -> None:
    records = _records()
    if not records:
        raise SystemExit("No coupling diagnostics found in solution/.")

    time = np.asarray([row["time"] for row in records], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=FIGURE_DPI, sharex=True)

    timing = axes[0, 0]
    vpm = _values(records, "timing_seconds", "vpm")
    fvm = _values(records, "timing_seconds", "fvm")
    transfer = sum(
        (_values(records, "timing_seconds", name) for name in ("vpm_bc", "transfer")),
        start=np.zeros_like(time),
    )
    timing.stackplot(
        time,
        vpm,
        fvm,
        transfer,
        labels=("VPM", "FVM", "transfer"),
        colors=(util.COLORS["vpm"], util.COLORS["fvm"], util.COLORS["accent"]),
        alpha=0.85,
    )
    timing.set(ylabel="wall time / step [s]", title="Coupling-step cost")
    timing.legend(loc="upper left", ncol=3, fontsize=7)

    population = axes[0, 1]
    population.plot(
        time,
        np.asarray([row.get("transfer_particle_count", 0) for row in records]) / 1e6,
        color=util.COLORS["vpm"],
        label="total",
    )
    for key, label, style in (
        ("n_existing", "existing", "-"),
        ("n_added", "correction", "--"),
    ):
        population.plot(
            time,
            _values(records, "transfer", key) / 1e6,
            linestyle=style,
            label=label,
        )
    population.set(ylabel="particles [million]", title="Particle population")
    population.legend(loc="upper left", fontsize=7)

    fidelity = axes[1, 0]
    sampled = _evaluated(records)
    t_sampled = time[sampled]
    marker = "o" if t_sampled.size < 60 else None
    fidelity.semilogy(
        t_sampled,
        np.maximum(_values(records, "transfer", "divergence_correction_l2")[sampled], 1e-16),
        color=util.COLORS["fvm"],
        marker=marker,
        markersize=3,
        label=r"correction $L_2$",
    )
    fidelity.set(xlabel="flow time [s]", ylabel="dimensionless", title=r"$\nabla\cdot\omega$")
    fidelity.legend(loc="upper right", fontsize=7)

    quality = axes[1, 1]
    quality.semilogy(
        time,
        np.maximum(_values(records, "transfer", "correction_vortex_strength_l1"), 1e-30),
        color=util.COLORS["accent"],
        label=r"$\Sigma|\Delta\Gamma|$",
    )
    quality.set(
        xlabel="flow time [s]",
        ylabel=r"vortex strength [m$^3$/s]",
        title="Applied local correction",
    )
    quality.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    util.save(fig, "coupling_diagnostics", figure_format, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    plot(args.format)


if __name__ == "__main__":
    main()
