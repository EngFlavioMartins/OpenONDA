#!/usr/bin/env python3
"""Plot coupled-run timing, particle population, and transfer diagnostics."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


from pathlib import Path
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import postprocess as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = util.FIGURE_DPI
FIGURE_HEIGHT_CM = 16.0
FIGURE_SIZE = util.figure_size(FIGURE_HEIGHT_CM)

# Manual layout controls (fractions of the fixed 12.5 cm canvas).
LAYOUT_LEFT = 0.14
LAYOUT_RIGHT = 0.86
LAYOUT_BOTTOM = 0.09
LAYOUT_TOP = 0.95
LAYOUT_HSPACE = 0.36
LEGEND_FONT_SIZE = util.FONT_SIZE_PT


def _records() -> list[dict]:
    path = util.SOLUTION / "coupler_diagnostics.jsonl"
    if not path.exists():
        return []
    records = []
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may leave one temporarily incomplete final line.
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return records


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    # Unevaluated diagnostics are null, not zero; NaN keeps them off the plot.
    return np.asarray(
        [
            np.nan if row.get(section, {}).get(key) is None else row.get(section, {}).get(key)
            for row in records
        ],
        dtype=float,
    )


def plot(figure_format: str, dpi: int = FIGURE_DPI) -> None:
    util._THEME.set_thesis_style()
    records = _records()
    if not records:
        raise SystemExit("No coupling diagnostics found in solution/.")

    time = np.asarray([row["time"] for row in records], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE, dpi=dpi, sharex=True)
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        bottom=LAYOUT_BOTTOM,
        top=LAYOUT_TOP,
        hspace=LAYOUT_HSPACE,
    )

    timing = axes[0]
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
    timing.set(ylabel="Wall time [s]", title="(a) Cost per coupling interval")
    timing.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        borderpad=0.3,
        labelspacing=0.25,
    )

    population = axes[1]
    population.plot(
        time,
        _values(records, "transfer", "n_particles_after") / 1e6,
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
    population.set(ylabel=r"$N$ [million]", title="(b) Particle population")
    population.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        borderpad=0.3,
        labelspacing=0.25,
    )

    fidelity = axes[2]
    state_change = np.sqrt(
        sum(
            _values(records, "transfer", f"state_change_vortex_strength_net_{axis}") ** 2
            for axis in "xyz"
        )
    )
    fidelity.plot(
        time,
        state_change,
        color=util.COLORS["fvm"],
    )
    fidelity.set(
        ylabel=r"$\|\Delta\sum_p\boldsymbol{\Gamma}_p\|$ [m$^3$/s]",
        title="(c) Net transfer state change",
    )

    quality = axes[3]
    for key, label, color in (
        ("replaced_vortex_strength_l1", "replaced", util.COLORS["fvm"]),
        ("injected_vortex_strength_l1", "injected", util.COLORS["accent"]),
    ):
        quality.plot(
            time,
            _values(records, "transfer", key),
            color=color,
            label=label,
        )
    quality.set(
        ylabel=r"$\sum_p\|\boldsymbol{\Gamma}_p\|$ [m$^3$/s]",
        title="(d) State replacement",
    )
    quality.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2, frameon=False)
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    axes[3].set_xlabel("Flow time [s]")
    util._THEME.fit_thesis_y_label_margins(fig, axes)

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
