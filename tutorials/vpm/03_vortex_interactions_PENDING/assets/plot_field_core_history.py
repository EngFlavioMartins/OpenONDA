#!/usr/bin/env python3
"""Plot projected field-core tracks without particle-group identities.

The input is a field-core history CSV with initially trailing/leading tracks
at 25%, 40% and 60% curl thresholds. Curves use 40%; shading spans the three
threshold estimates. Both stop at the first missing/ambiguous identity at any
threshold. An axial-order reversal is a sampled projected overtaking bracket,
not a three-dimensional breakdown criterion or a matched LBM comparison.

Validation and --check-only use only the standard library. Plotting imports
are deferred until explicitly rendering. R0 is supplied in metres; optional
tau = |Gamma| t / R0**2 requires circulation in m**2/s.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


THRESHOLDS = (0.25, 0.4, 0.6)
TRACKS = ("initial_trailing", "initial_leading")
LABELS = {"initial_trailing": "Initially trailing", "initial_leading": "Initially leading"}
REQUIRED = {
    "case",
    "step",
    "time",
    "threshold_fraction",
    "track",
    "x",
    "radius",
    "status",
    "reason",
}


def _finite(value, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number")
    return number


def _read(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = []
    reader = csv.DictReader(raw.decode("utf-8-sig").splitlines())
    if not REQUIRED.issubset(reader.fieldnames or ()):
        raise ValueError(f"{path}: missing required field-core columns")
    for number, row in enumerate(reader, start=2):
        context = f"{path}:{number}"
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{context}: incomplete CSV row")
        step = _finite(row["step"], context)
        if step < 0 or not step.is_integer():
            raise ValueError(f"{context}: invalid step")
        time = _finite(row["time"], context)
        threshold = _finite(row["threshold_fraction"], context)
        if time < 0 or threshold not in THRESHOLDS:
            raise ValueError(f"{context}: invalid time or unsupported threshold")
        if not row["case"] or not row["status"]:
            raise ValueError(f"{context}: missing case/status")
        parsed = {**row, "step": int(step), "time": time, "threshold_fraction": threshold}
        if row["status"] == "tracked":
            if row["track"] not in TRACKS:
                raise ValueError(f"{context}: tracked row lacks an initial identity")
            parsed["x"] = _finite(row["x"], context)
            parsed["radius"] = _finite(row["radius"], context)
            if parsed["radius"] <= 0:
                raise ValueError(f"{context}: nonpositive tracked radius")
        rows.append(parsed)
    if not rows:
        raise ValueError(f"{path}: empty field-core history")
    return rows, hashlib.sha256(raw).hexdigest()


def _prepare(rows: list[dict], *, r0: float, time_axis: str, circulation: float | None) -> dict:
    if not math.isfinite(r0) or r0 <= 0:
        raise ValueError("--r0 must be a positive radius in metres")
    if time_axis == "tau" and (
        circulation is None or not math.isfinite(circulation) or circulation == 0
    ):
        raise ValueError("--time-axis tau requires finite nonzero --circulation [m^2/s]")
    clock = {}
    frames = {}
    for row in rows:
        step, time = row["step"], row["time"]
        if step in clock and clock[step] != time:
            raise ValueError(f"Step {step}: conflicting physical clocks")
        clock[step] = time
        frames.setdefault(step, []).append(row)
    steps = sorted(frames)
    if any(clock[right] <= clock[left] for left, right in zip(steps, steps[1:])):
        raise ValueError("Physical time must increase with step")
    accepted = []
    stopped = None
    for step in steps:
        table = {}
        reasons = []
        for row in frames[step]:
            if row["status"] != "tracked":
                reasons.append(
                    f"{row['threshold_fraction']:.0%}: {row['status']} ({row['reason']})"
                )
                continue
            key = row["threshold_fraction"], row["track"]
            if key in table:
                raise ValueError(f"Step {step}: duplicate threshold/identity {key}")
            table[key] = row
        missing = {
            (threshold, track) for threshold in THRESHOLDS for track in TRACKS
        } - table.keys()
        if missing:
            reasons.append("missing threshold/initial-identity membership")
        if reasons:
            stopped = {"step": step, "time": clock[step], "reasons": reasons}
            break
        accepted.append({"step": step, "time": clock[step], "rows": table})
    if not accepted:
        raise ValueError("No initial interval with all threshold identities valid")
    if accepted[0]["step"] == 0:
        for threshold in THRESHOLDS:
            first = accepted[0]["rows"]
            if (
                first[threshold, "initial_trailing"]["x"]
                >= first[threshold, "initial_leading"]["x"]
            ):
                raise ValueError(
                    "Initial trailing/leading identity contradicts axial order at step zero"
                )
    scale = abs(circulation) / r0**2 if time_axis == "tau" else 1.0
    curves = {}
    for track in TRACKS:
        curves[track] = {}
        for quantity in ("x", "radius"):
            estimates = [
                [frame["rows"][threshold, track][quantity] / r0 for threshold in THRESHOLDS]
                for frame in accepted
            ]
            curves[track][quantity] = {
                "central": [values[1] for values in estimates],
                "lower": [min(values) for values in estimates],
                "upper": [max(values) for values in estimates],
            }
    brackets = {}
    for threshold in THRESHOLDS:
        for left, right in zip(accepted, accepted[1:]):

            def separation(frame):
                table = frame["rows"]
                return (
                    table[threshold, "initial_trailing"]["x"]
                    - table[threshold, "initial_leading"]["x"]
                )

            if separation(left) < 0 <= separation(right):
                brackets[str(threshold)] = {
                    "steps": [left["step"], right["step"]],
                    "time_seconds": [left["time"], right["time"]],
                    "axis_interval": [left["time"] * scale, right["time"] * scale],
                }
                break
    central = brackets.get("0.4")
    consensus = (
        central is not None
        and len(brackets) == len(THRESHOLDS)
        and all(value["steps"] == central["steps"] for value in brackets.values())
    )
    return {
        "steps": [frame["step"] for frame in accepted],
        "times_seconds": [frame["time"] for frame in accepted],
        "axis_values": [frame["time"] * scale for frame in accepted],
        "time_axis": time_axis,
        "r0_metres": r0,
        "circulation_m2_per_s": circulation,
        "curves": curves,
        "tracking_stop": stopped,
        "excluded_steps_from_stop": [
            step for step in steps if stopped is not None and step >= stopped["step"]
        ],
        "overtaking_brackets": brackets,
        "same_bracket_all_thresholds": consensus,
        "starts_at_initial_clock": accepted[0]["step"] == 0 and accepted[0]["time"] == 0,
        "cases": list(dict.fromkeys(row["case"] for row in rows)),
        "scope": "Projected field-core histories with fixed initial trailing/leading identities; threshold range is sensitivity, not statistical uncertainty. No three-dimensional breakdown or matched-LBM inference.",
    }


def _render(prepared: dict, output: Path, formats, dpi: int | None, thesis: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openonda import plotting as theme

    formats = formats or theme.EXPORT_FORMATS
    if any(value not in theme.EXPORT_FORMATS for value in formats):
        raise ValueError(f"Formats must be chosen from {theme.EXPORT_FORMATS}")
    (theme.set_thesis_style if thesis else theme.set_style)()
    colors = {
        "initial_trailing": theme.COLORS["FVMorange"],
        "initial_leading": theme.COLORS["VPMpurple"],
    }
    for figure_format in formats:
        fig, axes = plt.subplots(
            2, 1, figsize=theme.figure_size("stacked"), sharex=True, layout="constrained"
        )
        for track in TRACKS:
            for ax, quantity in zip(axes, ("x", "radius"), strict=True):
                values = prepared["curves"][track][quantity]
                ax.fill_between(
                    prepared["axis_values"],
                    values["lower"],
                    values["upper"],
                    color=colors[track],
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(
                    prepared["axis_values"],
                    values["central"],
                    color=colors[track],
                    marker="o",
                    label=LABELS[track],
                )
        axes[0].set_ylabel(r"$x/R_0$")
        axes[1].set_ylabel(r"$R/R_0$")
        axes[1].set_xlabel(
            "Time [s]" if prepared["time_axis"] == "time" else r"$\tau=|\Gamma|t/R_0^2$"
        )
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.ticklabel_format(useOffset=False, style="plain", axis="both")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
        central = prepared["overtaking_brackets"].get("0.4")
        if central is not None:
            low, high = central["axis_interval"]
            axes[0].axvspan(low, high, color=theme.COLORS["reference"], alpha=0.12, linewidth=0)
            qualifier = (
                "25–60% agree" if prepared["same_bracket_all_thresholds"] else "40% threshold only"
            )
            axes[0].text(
                0.02,
                0.98,
                "Projected overtaking bracket\n" + qualifier,
                transform=axes[0].transAxes,
                va="top",
                color=theme.COLORS["DarkText"],
            )
        stop = prepared["tracking_stop"]
        suffix = "\nStopped before ambiguous tracking" if stop is not None else ""
        axes[0].set_title("Projected field cores" + suffix)
        axes[1].set_title("40% curves; shading: 25–60% range")
        output.mkdir(parents=True, exist_ok=True)
        theme.save_fig(fig, output / "field_core_history", figure_format=figure_format, dpi=dpi)
    (output / "field_core_history.json").write_text(
        json.dumps(prepared, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Explicit field-core CSV; no other run data are read",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for PDF, PNG and provenance JSON"
    )
    parser.add_argument("--r0", type=float, required=True, help="Initial reference radius [m]")
    parser.add_argument("--time-axis", choices=("time", "tau"), default="time")
    parser.add_argument("--circulation", type=float, help="Circulation [m^2/s], required for tau")
    parser.add_argument("--formats", nargs="+", help="Shared formats; default PNG and PDF")
    parser.add_argument("--dpi", type=int, help="Override shared raster export resolution")
    parser.add_argument("--thesis", action="store_true", help="Use shared LaTeX thesis fonts")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Prepare/validate without plotting imports or writes",
    )
    args = parser.parse_args()
    try:
        if args.dpi is not None and args.dpi <= 0:
            raise ValueError("--dpi must be positive")
        if args.circulation is not None and not math.isfinite(args.circulation):
            raise ValueError("--circulation must be finite")
        rows, digest = _read(args.input)
        prepared = _prepare(
            rows, r0=args.r0, time_axis=args.time_axis, circulation=args.circulation
        )
        prepared.update(
            input_csv=str(args.input.resolve()), input_sha256=digest, input_rows=len(rows)
        )
        if args.check_only:
            print(
                json.dumps(
                    {key: value for key, value in prepared.items() if key != "curves"},
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return
        _render(prepared, args.output, args.formats, args.dpi, args.thesis)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
