#!/usr/bin/env python3
"""Plot accepted coupling costs and total particles."""

import argparse


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402

from openonda.plotting import CM, COLORS, centered_subplots_adjust, set_thesis_style  # noqa: E402

from openonda.results import read_json_lines

from . import postprocess as data  # noqa: E402

TIMING_PHASES = (
    (("vpm",), "VPM", COLORS["vpm"]),
    (("fvm",), "FVM", COLORS["fvm"]),
    (("vpm_boundary_condition", "transfer"), "coupling", COLORS["hybrid"]),
    (
        ("state_checks_and_samplers", "backup", "reporting", "coupling_control_and_wait"),
        "sampling and output",
        COLORS["gray"],
    ),
)


def _records() -> list[dict]:
    return read_json_lines(data.CASE_DIR / "solution/coupler_diagnostics.jsonl")


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    return np.asarray([row[section][key] for row in records], dtype=float)


def _timing_per_fvm_step(records: list[dict]) -> np.ndarray:
    """Normalize each exclusive recorded phase by its accepted FVM substeps."""
    substeps = np.asarray([row["n_fvm_substeps"] for row in records], dtype=float)
    phases = np.asarray(
        [
            sum(
                (_values(records, "timing_seconds", key) for key in keys),
                start=np.zeros(len(records)),
            )
            for keys, _, _ in TIMING_PHASES
        ]
    )
    return phases / substeps


def plot(figure_format: str) -> None:
    records = _records()
    time = np.asarray([row["time"] for row in records], dtype=float)
    costs = _timing_per_fvm_step(records)

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 9.5 * CM), sharex=True)
    axes[0].stackplot(
        time,
        *costs,
        labels=[label for _, label, _ in TIMING_PHASES],
        colors=[color for _, _, color in TIMING_PHASES],
        alpha=0.85,
    )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set(ylabel="Cost [s]", title="(a) Cost per FVM step (log scale)")
    axes[1].plot(time, _values(records, "transfer", "n_particles_after") / 1e6, color=COLORS["vpm"])
    axes[1].set(ylabel=r"$N$ [million]", title="(b) Total particles", xlabel="Flow time [s]")
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    cost_handles, cost_labels = axes[0].get_legend_handles_labels()
    legend_style = dict(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    figure.legend(
        cost_handles,
        cost_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        **legend_style,
    )
    centered_subplots_adjust(figure, outer=0.14, bottom=0.31, top=0.92, hspace=0.5)
    data.save_figure(figure, axes, "coupling_diagnostics", figure_format)
    print(
        f"Coupling diagnostics: {len(records)} accepted exchanges, "
        f"t={time[0]:g}–{time[-1]:g} s; full recorded cost per FVM step."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    arguments = parser.parse_args()
    plot(arguments.format)


if __name__ == "__main__":
    main()
