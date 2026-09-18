#!/usr/bin/env python3
"""Plot coupled-run timing, particle population, and transfer diagnostics."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

from . import postprocess as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = util.FIGURE_DPI
FIGURE_HEIGHT_CM = 10.5
FIGURE_SIZE = util.figure_size(FIGURE_HEIGHT_CM)

# Manual layout controls (fractions of the fixed 12.5 cm canvas).
LAYOUT_LEFT = 0.16
LAYOUT_RIGHT = 0.84
LAYOUT_BOTTOM = 0.13
LAYOUT_TOP = 0.92
LAYOUT_HSPACE = 0.42
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


def _timing_per_fvm_step(records: list[dict], key: str) -> np.ndarray:
    """Normalize one recorded coupling cost to one equivalent FVM step.

    A coupling record contains one VPM step, one transfer, and all FVM
    substeps needed to advance the same physical interval. Dividing each
    component by the recorded number of FVM substeps gives a common cost unit:
    wall seconds per FVM flow-field step. The normalization is performed per
    record so a run may use a different valid subcycling ratio in another
    configuration.

    Parameters
    ----------
    records : list[dict]
        Coupler diagnostic records containing ``timing_seconds`` and the
        positive integer ``n_fvm_substeps``.
    key : str
        Timing component, for example ``"vpm"``, ``"fvm"`` or ``"transfer"``.

    Returns
    -------
    numpy.ndarray
        Component wall time divided by the number of FVM substeps, in seconds
        per equivalent FVM step.

    Raises
    ------
    ValueError
        If a record has a missing, non-finite, or non-positive substep count.
    """
    timing = _values(records, "timing_seconds", key)
    try:
        substeps = np.asarray([row["n_fvm_substeps"] for row in records], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("each record must contain n_fvm_substeps") from error
    if (
        np.any(~np.isfinite(substeps))
        or np.any(substeps <= 0.0)
        or np.any(substeps != np.floor(substeps))
    ):
        raise ValueError("n_fvm_substeps must be a positive integer")
    return timing / substeps


def plot(figure_format: str, dpi: int = FIGURE_DPI) -> None:
    util._THEME.set_thesis_style()
    records = _records()
    if not records:
        raise SystemExit("No coupling diagnostics found in solution/.")

    time = np.asarray([row["time"] for row in records], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, dpi=dpi, sharex=True)
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        bottom=LAYOUT_BOTTOM,
        top=LAYOUT_TOP,
        hspace=LAYOUT_HSPACE,
    )

    timing = axes[0]
    vpm = _timing_per_fvm_step(records, "vpm")
    fvm = _timing_per_fvm_step(records, "fvm")
    transfer = sum(
        (_timing_per_fvm_step(records, name) for name in ("vpm_boundary_condition", "transfer")),
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
    timing.set(
        ylabel="Cost [s]",
        title="(a) Cost per FVM step",
    )
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
    population.plot(
        time,
        _values(records, "transfer", "n_particles_injected") / 1e6,
        linestyle="--",
        label="injected",
    )
    population.set(ylabel=r"$N$ [million]", title="(b) Particle population")
    population.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
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
        title="(c) Net change in total vortex strength",
    )

    fidelity.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    fidelity.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    axes[2].set_xlabel("Flow time [s]")
    util._THEME.fit_thesis_y_label_margins(fig, axes)
    # The longer cost label needs a slightly wider symmetric margin than the
    # automatic minimum to remain clear with the fixed thesis canvas.
    util._THEME.centered_subplots_adjust(fig, outer=LAYOUT_LEFT)

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
