#!/usr/bin/env python3
"""Compare cylinder forces over the actually available common interval."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    centered_subplots_adjust,
    set_thesis_style,
)

from . import postprocess as data

FORCE_COLUMNS = ("drag_coefficient", "lift_coefficient")
LEGACY_STARTUP_GAP = (2.0, 2.08)
LEGACY_STARTUP_IMPULSE_TIME = 2.04


def _legacy_startup_gap(time, coupled, policy):
    """Identify only the recorded old startup impulse, without changing samples.

    The common plotting grid includes interpolated values between native force
    samples. Gap that entire interpolation interval around the one bad native
    lift sample, retaining both neighboring samples. NaNs break the line rather
    than drawing an invented bridge. Fresh smooth-startup histories are excluded.
    """
    mask = np.zeros(len(time), dtype=bool)
    recorded_policy = {
        "schema": "openonda-cylinder-startup/1",
        "startup_duration": 2.0,
        "exchange_time_step": 0.04,
        "fvm_time_step": 0.008,
        "switch_step": 50,
        "startup_freestream_velocity": [1.0, 0.1, 0.0],
        "steady_freestream_velocity": [1.0, 0.0, 0.0],
    }
    if any(policy.get(key) != value for key, value in recorded_policy.items()):
        return mask, {"enabled": False, "reason": "not the recorded legacy startup policy"}
    native = coupled.loc[np.isclose(coupled.time, LEGACY_STARTUP_IMPULSE_TIME,
                                    rtol=0, atol=1e-10)]
    if len(native) != 1:
        return mask, {"enabled": False, "reason": "legacy impulse sample unavailable"}
    start, end = LEGACY_STARTUP_GAP
    mask = (time > start + 1e-10) & (time < end - 1e-10)
    return mask, {
        "enabled": bool(mask.any()),
        "column": "lift_coefficient",
        "source": "coupled",
        "reason": "isolated pressure impulse from the original abrupt transverse-velocity switch",
        "recorded_startup_schema": policy["schema"],
        "open_display_gap_seconds": [start, end],
        "omitted_native_time": float(native.time.iloc[0]),
        "omitted_native_lift_coefficient": float(native.lift_coefficient.iloc[0]),
        "omitted_common_grid_times": time[mask].tolist(),
        "raw_history_modified": False,
        "error_statistics_include_omitted_sample": True,
        "restoration_option": "--include-startup-impulse",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--include-startup-impulse", action="store_true",
                        help="Show the recorded legacy startup lift impulse as measured.")
    arguments = parser.parse_args()

    coupled = data.history(data.CASE_DIR / "samples" / "forces_history.csv", FORCE_COLUMNS)
    reference = data.history(data.reference_directory() / "forces_history.csv", FORCE_COLUMNS)
    time, coupled_values, reference_values, errors = data.common_history(
        coupled, reference, FORCE_COLUMNS
    )
    coupled_display = coupled_values.copy()
    omission = {"enabled": False, "reason": "raw display requested"}
    if not arguments.include_startup_impulse:
        policy_path = data.CASE_DIR / "solution/cylinder_startup.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
        mask, omission = _legacy_startup_gap(time, coupled, policy)
        coupled_display[mask, 1] = np.nan
    data.write_json(
        "reference_force_errors.json",
        {
            "reference": str(data.reference_directory().relative_to(data.CASE_DIR)),
            "reference_scope": "single-mesh comparison; not a grid-independence claim",
            "time_interval": [float(time[0]), float(time[-1])],
            "errors": errors,
            "display_only_startup_omission": omission,
            **data.history_coverage(coupled, reference),
        },
    )
    print(
        f"Force comparison: common saved coverage {time[0]:g}–{time[-1]:g} s; "
        "interpolated only within that interval, without time shifts."
    )
    if omission["enabled"]:
        print("Force display only: legacy Cl impulse at 2.04 s omitted with a gap; "
              "raw force samples and quantitative errors are unchanged. "
              "Use --include-startup-impulse to show it.")

    set_thesis_style()
    height = 10.5 if omission["enabled"] else 8.3
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, height * CM), sharex=True)
    labels = (r"$C_D$", r"$C_L$")
    for index, axis in enumerate(axes):
        axis.plot(
            time,
            reference_values[:, index],
            color=COLORS["reference"],
            linewidth=REFERENCE_LINE_WIDTH,
            label="Reference FVM",
            linestyle="--",
        )
        axis.plot(
            time,
            coupled_display[:, index],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled FVM",
        )
        axis.set_ylabel(labels[index])
        axis.grid(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.14 if omission["enabled"] else 0.025),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
    )
    if omission["enabled"]:
        figure.text(0.5, 0.012,
                    r"Legacy $C_L$ impulse omitted at $t=2.04$ s."
                    "\nRaw data and errors unchanged.",
                    ha="center", va="bottom")
    axes[1].set_xlabel(r"$tU_\infty/D$")
    centered_subplots_adjust(figure, outer=0.135,
                             bottom=0.38 if omission["enabled"] else 0.28,
                             top=0.93, hspace=0.12)
    data.save_figure(figure, axes, "reference_forces", arguments.format)


if __name__ == "__main__":
    main()
