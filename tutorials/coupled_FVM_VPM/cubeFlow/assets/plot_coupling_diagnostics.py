#!/usr/bin/env python3
"""Plot coupled-run timing, particle population, and handoff diagnostics."""

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
    return np.asarray([row.get(section, {}).get(key, 0.0) for row in records], dtype=float)


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
        (_values(records, "timing_seconds", name) for name in ("donor", "fringe", "handoff")),
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
        np.asarray([row.get("handoff_particle_count", 0) for row in records]) / 1e6,
        color=util.COLORS["vpm"],
        label="total",
    )
    for key, label, style in (
        ("n_remesh_out", "overlap/buffer", "-"),
        ("n_free", "free wake", "--"),
    ):
        population.plot(
            time,
            _values(records, "handoff", key) / 1e6,
            linestyle=style,
            label=label,
        )
    population.set(ylabel="particles [million]", title="Particle population")
    population.legend(loc="upper left", fontsize=7)

    pruning = axes[1, 0]
    pruning.plot(
        time,
        _values(records, "handoff", "n_pruned") / 1e3,
        color=util.COLORS["fvm"],
        label="base prune",
    )
    shell = _values(records, "handoff", "n_overlap_shell_pruned") / 1e3
    if np.any(shell):
        pruning.plot(time, shell, color=util.COLORS["vpm"], label="non-outflow shell")
    pruning.set(xlabel="flow time [s]", ylabel="particles [thousand]", title="Handoff pruning")
    pruning.legend(loc="upper right", fontsize=7)

    quality = axes[1, 1]
    base_loss = 100.0 * _values(records, "handoff", "pruned_circulation_fraction")
    shell_loss = 100.0 * _values(records, "handoff", "overlap_shell_pruned_circulation_fraction")
    quality.semilogy(time, np.maximum(base_loss, 1e-8), label=r"all pruned $\Sigma|\Gamma|$")
    if np.any(shell_loss):
        quality.semilogy(
            time,
            np.maximum(shell_loss, 1e-8),
            label=r"shell-pruned $\Sigma|\Gamma|$",
        )
    quality.set(
        xlabel="flow time [s]",
        ylabel="circulation fraction [%]",
        title="Pruning strength",
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
