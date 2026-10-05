#!/usr/bin/env python3
"""Plot accepted coupling costs and total particles."""

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
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402

from openonda.plotting import CM, COLORS, centered_subplots_adjust, set_thesis_style  # noqa: E402

from . import postprocess as data  # noqa: E402

TIMING_PHASES = (
    (("vpm",), "VPM", COLORS["vpm"]),
    (("fvm",), "FVM", COLORS["fvm"]),
    (("vpm_boundary_condition", "transfer"), "coupling", COLORS["hybrid"]),
    (("health_and_samplers", "backup", "reporting", "orchestration_and_wait"),
     "sampling and output", COLORS["gray"]),
)

CONSERVATION_CHECKS = (
    ("renewal_conservation_error", "renewal_vortex_strength_tolerance",
     "vortex strength", COLORS["vpm"]),
    ("renewal_linear_impulse_error", "renewal_linear_impulse_tolerance",
     "linear impulse", COLORS["fvm"]),
)


def _records() -> list[dict]:
    path = data.CASE_DIR / "solution/coupler_diagnostics.jsonl"
    records = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # The running solver may still be writing the final record.
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise
    return records


def _values(records: list[dict], section: str, key: str) -> np.ndarray:
    # Unevaluated diagnostics are null; leave a gap instead of inventing zero.
    return np.asarray([
        np.nan if row.get(section, {}).get(key) is None else row[section][key]
        for row in records
    ], dtype=float)


def _timing_per_fvm_step(records: list[dict]) -> np.ndarray:
    """Return exclusive phase costs divided by each record's FVM substep count."""
    try:
        substeps = np.asarray([row["n_fvm_substeps"] for row in records], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("each record must contain n_fvm_substeps") from error
    if (np.any(~np.isfinite(substeps)) or np.any(substeps <= 0)
            or np.any(substeps != np.floor(substeps))):
        raise ValueError("n_fvm_substeps must be a positive integer")

    # evolution_total is an aggregate and donor_gather is already inside FVM.
    # Include only exclusive phases so the stack recovers the recorded total.
    components = {key: _values(records, "timing_seconds", key)
                  for keys, _, _ in TIMING_PHASES for key in keys}
    total = _values(records, "timing_seconds", "total")
    exclusive = np.asarray(list(components.values()))
    if np.any(~np.isfinite(exclusive)) or np.any(exclusive < 0) or np.any(~np.isfinite(total)):
        raise ValueError("each record must contain finite, nonnegative exclusive phase timings")
    phases = np.asarray([
        sum((components[key] for key in keys), start=np.zeros(len(records)))
        for keys, _, _ in TIMING_PHASES
    ])
    if not np.allclose(phases.sum(axis=0), total, rtol=1e-10, atol=1e-9):
        raise ValueError("exclusive phase timings do not sum to the recorded total")
    return phases / substeps


def _conservation_fraction(records: list[dict], error_key: str, tolerance_key: str) -> np.ndarray:
    """Return the applied renewal residual as a fraction of its accepted tolerance.

    These checks compare the renewed particles, after casting to storage
    precision, with their pre-pruning blended target. They do not measure the
    physical change from the FVM exchange or a whole-domain conservation law.
    Missing checks and the zero/zero placeholder for no renewal remain gaps.
    """
    error = _values(records, "transfer", error_key)
    tolerance = _values(records, "transfer", tolerance_key)
    if (np.any(np.isinf(error)) or np.any(np.isinf(tolerance))
            or np.any(error < 0) or np.any(tolerance < 0)
            or np.any((tolerance == 0) & (error > 0))):
        raise ValueError("renewal conservation errors and tolerances must be finite and nonnegative")
    return np.divide(error, tolerance, out=np.full(len(records), np.nan),
                     where=np.isfinite(error) & np.isfinite(tolerance) & (tolerance > 0))


def plot(figure_format: str) -> None:
    records = _records()
    if not records:
        raise ValueError("No coupling diagnostics found in solution/.")
    time = np.asarray([row["time"] for row in records], dtype=float)
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("coupling times must be finite and strictly increasing")
    costs = _timing_per_fvm_step(records)

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 9.5 * CM), sharex=True)
    axes[0].stackplot(
        time, *costs,
        labels=[label for _, label, _ in TIMING_PHASES],
        colors=[color for _, _, color in TIMING_PHASES],
        alpha=0.85,
    )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set(ylabel="Cost [s]", title="(a) Cost per FVM step (log scale)")
    axes[1].plot(time, _values(records, "transfer", "n_particles_after") / 1e6,
                 color=COLORS["vpm"])
    axes[1].set(ylabel=r"$N$ [million]", title="(b) Total particles", xlabel="Flow time [s]")
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    cost_handles, cost_labels = axes[0].get_legend_handles_labels()
    legend_style = dict(frameon=True, fancybox=True, framealpha=0.9, facecolor="white",
                        edgecolor="0.8", handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    figure.legend(cost_handles, cost_labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                  ncol=2, **legend_style)
    centered_subplots_adjust(figure, outer=0.14, bottom=0.31, top=0.92, hspace=0.5)
    data.save_figure(figure, axes, "coupling_diagnostics", figure_format)
    print(f"Coupling diagnostics: {len(records)} accepted exchanges, "
          f"t={time[0]:g}–{time[-1]:g} s; full recorded cost per FVM step.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    arguments = parser.parse_args()
    plot(arguments.format)


if __name__ == "__main__":
    main()
