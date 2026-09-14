#!/usr/bin/env python3
"""Plot independently verified forces and profiles from a fixed 3D prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays

REFERENCE, HYBRID, PARTICLES = "#333333", "#156d91", "#ad572f"


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.verification.read_text())
    assert report["status"] == "complete" and report["schema"] == "openonda-long-wake-prefix-verification-3d/2"
    for row in report["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    geometry = read_arrays(ROOT / report["profile_geometry"]["path"])
    points, fluid, small = (geometry[key] for key in ("position", "fluid_mask", "small_mask"))
    history, frames = report["comparison_history"], report["profile_frames"]
    time = np.array([r["physical_time"] for r in history])
    profile_time = np.array([r["physical_time"] for r in frames])
    full_force = np.array([r["forces"]["full"]["coefficient_vector"] for r in frames])
    hybrid_force = np.array([r["forces"]["hybrid"]["coefficient_vector"] for r in frames])
    start, stop = report["physical_time_bounds"]
    title = f"Fully 3D matched cube · verified prefix t = {start:g}–{stop:g}"
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "savefig.facecolor": "white"})
    outputs = []

    def save(figure, name):
        for extension in ("png", "svg"):
            path = args.output / f"{name}.{extension}"
            figure.savefig(path, dpi=175)
            outputs.append(hash_file(path))
        plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11.5, 8.2), sharex=True, layout="constrained")
    for axis, label in enumerate(("Drag", "Transverse force y", "Transverse force z")):
        t = time if axis == 0 else profile_time
        full = np.array([r["full_drag_coefficient"] for r in history]) if axis == 0 else full_force[:, axis]
        hybrid = np.array([r["hybrid_drag_coefficient"] for r in history]) if axis == 0 else hybrid_force[:, axis]
        marker = None if axis == 0 else "o"
        axes[axis, 0].plot(t, full, color=REFERENCE, linestyle="--", label="Full FVM", marker=marker, markersize=3)
        axes[axis, 0].plot(t, hybrid, color=HYBRID, label="Hybrid", marker=marker, markersize=3)
        axes[axis, 1].plot(t, hybrid - full, color=HYBRID, marker=marker, markersize=3)
        axes[axis, 1].axhline(0, color=".5", linewidth=.8)
        component = "xyz"[axis]
        axes[axis, 0].set(title=label, ylabel=f"$C_{component}$")
        axes[axis, 1].set(title="Hybrid − full FVM", ylabel=f"$\u0394 C_{component}$")
        for ax in axes[axis]:
            ax.grid(alpha=.2)
            ax.set_xlim(start, stop)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Physical flow time")
    figure.suptitle(title + "\nDrag sampled every 0.05; all force components checked from saved states every 0.25", fontsize=12)
    save(figure, "forces")

    selected = [r for r in frames if r["coupling_step"] in args.profile_steps]
    assert len(selected) == len(set(args.profile_steps))
    figure, axes = plt.subplots(len(selected), 2, figsize=(11.5, 2.65 * len(selected)), sharex=True, sharey=True, squeeze=False, layout="constrained")
    all_limits = []
    for index, row in enumerate(selected):
        field = read_arrays(ROOT / row["profile_fields"]["path"])
        for line_y, column, name in ((0., 0, "Centreline: y/D = 0"), (.75, 1, "Off-axis: y/D = 0.75")):
            ax = axes[index, column]
            line = points[:, 1] == line_y
            for key, mask, color, style, label in (("full_profile", fluid, REFERENCE, "--", "Full FVM"),
                                                   ("small_profile", small, HYBRID, "-", "Hybrid FVM"),
                                                   ("vpm_profile", fluid & ~small, PARTICLES, "-", "VPM outside FVM")):
                values = np.full(len(points), np.nan)
                if key == "vpm_profile":
                    values[fluid] = field[key][:, 0]
                    values[~mask] = np.nan
                else:
                    values[mask] = field[key][:, 0]
                ax.plot(points[line, 0], values[line], color=color, linestyle=style, label=label, linewidth=1.6)
                all_limits.extend(values[np.isfinite(values)].tolist())
            ax.axvspan(-1.5, 1.5, color=".9", alpha=.45)
            if line_y == 0:
                ax.axvspan(-.5, .5, color=".65", alpha=.7)
            ax.set(title=f"{name} · t = {row['physical_time']:g}", ylabel="$u_x/U_\u221e$", xlim=(-3, 10))
            ax.grid(alpha=.16)
    lower, upper = min(all_limits), max(all_limits)
    axes[0, 0].set_ylim(lower - .04 * (upper - lower), upper + .04 * (upper - lower))
    axes[0, 0].legend(loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("x/D")
    figure.suptitle(title + "\nSame-time profiles; grey band is the small FVM region; body points are excluded", fontsize=12)
    save(figure, "profiles")

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), layout="constrained")
    for key, label, color in (("fvm_velocity_rms_over_Uinf", "Whole small FVM", HYBRID),
                               ("fvm_near_body_velocity_rms_over_Uinf", "Near-body FVM", "#2f856b"),
                               ("vpm_sampled_velocity_rms_over_Uinf", "VPM sample inside FVM", PARTICLES)):
        axes[0, 0].plot(time, [100 * r[key] for r in history], label=label, color=color)
    axes[0, 0].set(title="Volume-weighted vector velocity error", ylabel="RMS error / U∞ [%]")
    for line, ax, title_line in (("centreline", axes[0, 1], "Centreline outside FVM"),
                                  ("offaxis_y075", axes[1, 1], "Off-axis outside FVM: y/D = 0.75")):
        for region, label, color in (("upstream_exterior", "Upstream: x/D < −1.5", "#777777"),
                                      ("near_wake_exterior", "Near wake: 1.5 < x/D ≤ 4", PARTICLES),
                                      ("far_wake_exterior", "Far wake: x/D > 4", "#8878a5")):
            values = [100 * r["metrics"][f"{line}_{region}_vpm"]["vector_rms_over_Uinf"] for r in frames]
            ax.plot(profile_time, values, label=label, color=color, marker="o", markersize=3)
        ax.set(title=title_line, ylabel="Line vector RMS / U∞ [%]")
    for line, name, color in (("centreline", "Centreline", HYBRID), ("offaxis_y075", "Off-axis", "#8878a5")):
        for kind, style, qualifier in (("raw", "--", "full reference stencil"), ("matched_sampling", "-", "matched FVM stencil")):
            values = [100 * r["metrics"][f"{line}_composite_{kind}"]["vector_rms_over_Uinf"] for r in frames]
            axes[1, 0].plot(profile_time, values, color=color, linestyle=style, label=f"{name}, {qualifier}")
    axes[1, 0].set(title="Physical hybrid profile: FVM inside, VPM outside", ylabel="Line vector RMS / U∞ [%]")
    for ax in axes.ravel():
        ax.set_xlabel("Physical flow time")
        ax.set_xlim(start, stop)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    figure.suptitle(title + "\nField and sampling errors are retained separately; no phase shift or fitted scale", fontsize=12)
    save(figure, "velocity-errors")

    direct_regions = []
    for row in frames:
        check = row["direct_profile_check"]
        if not check:
            continue
        field = read_arrays(ROOT / check["fields"]["path"])
        p = field["position"]
        record = {"physical_time": row["physical_time"], "regions": {}}
        for line_y, name in ((0., "centreline"), (.75, "offaxis_y075")):
            mask = (p[:, 1] == line_y) & (p[:, 0] > 1.5) & (p[:, 0] <= 4)
            def value(a, selected=mask):
                return float(np.sqrt(np.mean(np.sum(a[selected]**2, axis=1))))
            record["regions"][name] = {
                "direct_minus_saved_query_rms": value(field["direct_velocity"] - field["saved_query_velocity"]),
                "saved_query_minus_reference_rms": value(field["saved_query_velocity"] - field["full_profile"]),
                "direct_minus_reference_rms": value(field["direct_velocity"] - field["full_profile"]),
            }
        direct_regions.append(record)
    own = Path(__file__).resolve()
    archived = args.output / "sources" / own.relative_to(ROOT)
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(own.read_bytes())
    result = {"schema": "openonda-wake-prefix-figures-3d/1", "status": "complete", "spatial_dimensions": 3,
              "verification": hash_file(args.verification), "sources": [hash_file(own)], "figures": outputs,
              "near_wake_direct_comparisons": direct_regions,
              "limitations": ["These plots cover the verified accepted prefix only. They do not imply completed long-run or developed-wake validation."]}
    (args.output / "wake-prefix-figures-3d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"figures": [row["path"] for row in outputs], "direct_comparisons": direct_regions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, nargs="+", default=[20, 40, 70])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.verification, args.output = args.verification.resolve(), args.output.resolve()
    run(args)
