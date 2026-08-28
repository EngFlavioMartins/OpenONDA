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

FIGURE_FORMAT = "png"
FIGURE_DPI = util.FIGURE_DPI
FIGURE_HEIGHT_CM = 10.0
FIGURE_SIZE = util.figure_size(FIGURE_HEIGHT_CM)

# Manual layout controls (fractions of the fixed 12.5 cm canvas).
LAYOUT_LEFT = 0.15
LAYOUT_RIGHT = 0.98
LAYOUT_BOTTOM = 0.11
LAYOUT_TOP = 0.95
LAYOUT_WSPACE = 0.58
LAYOUT_HSPACE = 0.42
LEGEND_FONT_SIZE = util.FONT_SIZE_PT


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


def plot(figure_format: str, dpi: int = FIGURE_DPI) -> None:
    records = _records()
    if not records:
        raise SystemExit("No coupling diagnostics found in solution/.")

    time = np.asarray([row["time"] for row in records], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=dpi, sharex=True)
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        bottom=LAYOUT_BOTTOM,
        top=LAYOUT_TOP,
        wspace=LAYOUT_WSPACE,
        hspace=LAYOUT_HSPACE,
    )

    timing = axes[0, 0]
    vpm = _values(records, "timing_seconds", "vpm")
    fvm = _values(records, "timing_seconds", "fvm")
    transfer = sum(
        (
            _values(records, "timing_seconds", name)
            for name in ("vpm_boundary_condition", "transfer")
        ),
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
    timing.set(ylabel="s/step", title="Step cost")
    timing.legend(
        loc="upper left",
        ncol=1,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        borderpad=0.3,
        labelspacing=0.25,
    )

    population = axes[0, 1]
    population.plot(
        time,
        np.asarray([row.get("n_transfer_particles", 0) for row in records]) / 1e6,
        color=util.COLORS["vpm"],
        label="total",
    )
    for key, label, style in (
        ("n_particles_retained", "retained", "-"),
        ("n_particles_injected", "injected", "--"),
    ):
        population.plot(
            time,
            _values(records, "transfer", key) / 1e6,
            linestyle=style,
            label=label,
        )
    population.set(ylabel=r"$N$ [million]", title="Particles")
    population.legend(
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        borderpad=0.3,
        labelspacing=0.25,
    )

    fidelity = axes[1, 0]
    state_change = np.sqrt(
        sum(
            _values(records, "transfer", f"state_change_vortex_strength_net_{axis}") ** 2
            for axis in "xyz"
        )
    )
    fidelity.semilogy(
        time,
        np.maximum(state_change, 1e-30),
        color=util.COLORS["fvm"],
    )
    fidelity.set(
        xlabel="flow time [s]",
        ylabel=r"$|\Delta\Gamma|$ [m$^3$/s]",
        title="Net state change",
    )

    quality = axes[1, 1]
    for key, label, color in (
        ("replaced_vortex_strength_l1", "replaced", util.COLORS["fvm"]),
        ("injected_vortex_strength_l1", "injected", util.COLORS["accent"]),
    ):
        quality.semilogy(
            time,
            np.maximum(_values(records, "transfer", key), 1e-30),
            color=color,
            label=label,
        )
    quality.set(
        xlabel="flow time [s]",
        ylabel=r"$\|\Gamma\|_1$ [m$^3$/s]",
        title="State replacement",
    )
    quality.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)

    util.save(fig, "coupling_diagnostics", figure_format, dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default=FIGURE_FORMAT)
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="PNG resolution in dpi.")
    args = parser.parse_args()
    plot(args.format, args.dpi)


if __name__ == "__main__":
    main()
