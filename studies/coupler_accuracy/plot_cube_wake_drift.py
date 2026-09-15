"""Render the preserved 3D cube audit and controlled transfer comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from openonda import plotting as theme


def read(path):
    return json.loads(path.read_text())


def save(fig, axes, output, name, extension):
    theme.fit_thesis_y_label_margins(fig, axes)
    theme.validate_thesis_figure(fig, axes)
    fig.savefig(output / f"{name}.{extension}", dpi=theme.DEFAULT_DPI, bbox_inches=None)
    plt.close(fig)


def profiles(args, time, directory):
    """Plot saved line samples without smoothing or inventing FVM coverage."""
    fig, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 15.5 * theme.CM))
    theme.centered_subplots_adjust(fig, outer=0.18, bottom=0.1, top=0.79, hspace=0.55)
    for ax, name, label in zip(
        axes,
        ("centreline", "offaxis_y075"),
        (r"(a) $y/D=0$", r"(b) $y/D=0.75$"),
        strict=True,
    ):
        values = pd.read_csv(directory / f"{name}_t{time:g}.csv")
        x = values["position_x"].to_numpy()
        inside = (x >= -1.5) & (x <= 3.0)
        for key, title, colour, style in (
            ("reference", "Reference FVM", "RefGray", "-"),
            ("baseline_vpm", "Baseline VPM", "FVMorange", "-"),
            ("candidate_vpm", "Corrected VPM", "TUDcyan", "-"),
            ("candidate", "Corrected FVM", "TUDcyan", "--"),
        ):
            y = values[f"{key}_velocity_x"].to_numpy()
            # Show the actual samples; the joining lines are not extra data.
            ax.plot(
                x[inside],
                y[inside],
                color=theme.COLORS[colour],
                ls=style,
                marker=".",
                markersize=2,
                label=title,
            )
        ax.axvline(1.5, color=theme.COLORS["DarkText"], ls=":", lw=0.8)
        ax.axhline(0, color=theme.COLORS["RefGray"], lw=0.6, zorder=0)
        if name == "centreline":
            ax.axvspan(-0.5, 0.5, facecolor=theme.COLORS["RefGray"], alpha=0.15, zorder=0)
        ax.set(
            xlabel=r"$x/D$",
            ylabel=r"$u_x/U_\infty$",
            xlim=(-1.5, 3),
            xticks=[-1, 0, 1, 2, 3],
            title=label,
        )
        ax.yaxis.set_major_locator(MaxNLocator(4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False
    )
    fig.text(0.5, 0.855, rf"$tU_\infty/D={time:g}$; dotted line: FVM $x_{{\max}}$", ha="center")
    save(fig, axes, args.output, f"velocity_profiles_t{time:g}", args.format)


def forces(args, files):
    """Show the longest supplied complete matched force interval."""
    if not files:
        return
    histories = [pd.read_csv(path) for path in files]
    data = max(histories, key=lambda d: d["time"].iloc[-1] - d["time"].iloc[0])
    if len(data) < 2:
        return
    fig, ax = plt.subplots(figsize=(12.5 * theme.CM, 10.5 * theme.CM))
    theme.centered_subplots_adjust(fig, outer=0.19, bottom=0.14, top=0.73)
    for name, label, colour, style in (
        ("reference", "Reference FVM", "RefGray", "-"),
        ("baseline", "Baseline FVM", "FVMorange", "-"),
        ("candidate", "Corrected FVM", "TUDcyan", "--"),
    ):
        ax.plot(data["time"], data[name], color=theme.COLORS[colour], ls=style, label=label)
    ax.set(xlabel=r"$tU_\infty/D$", ylabel=r"$C_D$")
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    save(fig, (ax,), args.output, "matched_drag_history", args.format)


def vpm_outflow_error(args, rows):
    """Expose the remaining vector error at the actual saved wake-plane probes."""
    if not rows:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 14.5 * theme.CM))
    theme.centered_subplots_adjust(fig, outer=0.19, bottom=0.11, top=0.81, hspace=0.65)
    for name, label, colour, marker in (
        ("baseline", "Baseline VPM", "FVMorange", "o"),
        ("candidate", "Corrected VPM", "TUDcyan", "s"),
    ):
        data = [r["sampled_plane"][name]["near_xmax"] for r in rows]
        times = [r["time"] for r in rows]
        axes[0].plot(
            times,
            [100 * r["rms"] for r in data],
            color=theme.COLORS[colour],
            marker=marker,
            label=label,
        )
        axes[1].plot(
            times,
            [100 * np.linalg.norm(r["component_rms"][1:]) for r in data],
            color=theme.COLORS[colour],
            marker=marker,
        )
    for ax, title in zip(
        axes,
        ("(a) All three velocity components", "(b) Transverse velocity components"),
        strict=True,
    ):
        ax.set(xlabel=r"$tU_\infty/D$", ylabel=r"RMS error (\% $U_\infty$)", title=title)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.text(0.5, 0.89, r"$z=0$; $1.25 \le x/D \le 1.75$", ha="center")
    save(fig, axes, args.output, "vpm_outflow_velocity_error", args.format)


def plot(args):
    args.output.mkdir(parents=True, exist_ok=True)
    theme.set_thesis_style()
    old = read(args.results / "audit/audit.json")["times"]
    phase = read(args.results / "lattice-phase-production/phase.json")["times"]
    by_time = {}
    profile_directories = {}
    force_files = []
    for filename in args.comparisons:
        history = filename.parent / "drag_history.csv"
        if history.exists():
            force_files.append(history)
        for row in read(filename)["times"]:
            if row["time"] in by_time and by_time[row["time"]] != row:
                raise ValueError("Conflicting comparison results at the same time")
            by_time[row["time"]] = row
            profile_directories[row["time"]] = filename.parent
    new = [by_time[t] for t in sorted(by_time)]
    old_by_time = {r["time"]: r for r in old}
    for row in new:
        old_by_time[row["time"]] = {
            "time": row["time"],
            "native_3d": {"whole": {"velocity": row["native_3d"]["baseline"]["whole"]}},
            "reattachment_x": {
                "vpm": row["profiles"]["reattachment_x"]["baseline_vpm"],
                "reference": row["profiles"]["reattachment_x"]["reference"],
            },
        }
    old = [old_by_time[t] for t in sorted(old_by_time)]
    forces(args, force_files)
    vpm_outflow_error(args, new)
    orange, blue, gray = (theme.COLORS[k] for k in ("FVMorange", "TUDcyan", "RefGray"))

    fig, axes = plt.subplots(2, 1, figsize=(12.5 * theme.CM, 15.5 * theme.CM))
    theme.centered_subplots_adjust(fig, outer=0.18, bottom=0.1, top=0.8, hspace=0.55)
    t = [r["time"] for r in old]
    axes[0].plot(
        t,
        [100 * r["native_3d"]["whole"]["velocity"]["rms"] for r in old],
        color=orange,
        marker="o",
        label="Baseline",
    )
    axes[1].plot(t, [r["reattachment_x"]["vpm"] for r in old], color=orange, marker="o")
    if new:
        axes[0].plot(
            [r["time"] for r in new],
            [100 * r["native_3d"]["candidate"]["whole"]["rms"] for r in new],
            color=blue,
            marker="s",
            label="Centred lattice",
        )
        axes[1].plot(
            [r["time"] for r in new],
            [r["profiles"]["reattachment_x"]["candidate_vpm"] for r in new],
            color=blue,
            marker="s",
        )
    (reference_line,) = axes[1].plot(
        t, [r["reattachment_x"]["reference"] for r in old], color=gray, ls="--", label="Reference"
    )
    axes[0].set(ylabel=r"RMS error (\% $U_\infty$)", title="(a) Native 3D velocity comparison")
    axes[1].set(ylabel=r"$x_R/D$", title="(b) Centreline reattachment")
    axes[1].axhline(1.5, color=gray, ls=":", lw=0.8)
    axes[1].text(10, 1.53, r"FVM $x_{\max}$", ha="center", va="bottom")
    for ax in axes:
        ax.set(xlabel=r"$tU_\infty/D$", xlim=(0, 15), xticks=[0, 5, 10, 15])
        ax.yaxis.set_major_locator(MaxNLocator(4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        [*handles, reference_line],
        [*labels, "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    save(fig, axes, args.output, "wake_error_and_reattachment", args.format)

    if new:
        fig, ax = plt.subplots(figsize=(12.5 * theme.CM, 10.5 * theme.CM))
        theme.centered_subplots_adjust(fig, outer=0.19, bottom=0.14, top=0.78)
        for name, label, color in (
            ("baseline", "Baseline", orange),
            ("candidate", "Corrected", blue),
        ):
            ax.plot(
                [r["time"] for r in new],
                [100 * r["native_3d"][name]["wake_boundary"]["rms"] for r in new],
                color=color,
                marker="o",
                label=label,
            )
        ax.set(
            xlabel=r"$tU_\infty/D$",
            ylabel=r"RMS error (\% $U_\infty$)",
            title=r"Downstream strip: $1.25 \le x/D \le 1.5$",
        )
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
        save(fig, (ax,), args.output, "downstream_velocity_error", args.format)
        for time in (6.0, 8.0):
            if np.any(np.isclose(list(by_time), time, atol=1e-8, rtol=0)):
                profiles(args, time, profile_directories[time])

    fig, ax = plt.subplots(figsize=(12.5 * theme.CM, 10.5 * theme.CM))
    theme.centered_subplots_adjust(fig, outer=0.19, bottom=0.14, top=0.75)
    for name, label, color, style, marker in (
        ("minimum_corner", "Lower corner", orange, "-", "o"),
        ("mirrored_transverse", "Mirrored corner", gray, "--", "+"),
        ("centred_all", "Body centre", blue, "-", "s"),
    ):
        ax.semilogy(
            [r["time"] for r in phase],
            [r["phases"][name][0]["rotation_defect_rms"] for r in phase],
            label=label,
            color=color,
            ls=style,
            marker=marker,
        )
    ax.set(
        xlabel=r"Reference state $tU_\infty/D$",
        ylabel=r"Rotation defect / $U_\infty$",
        xticks=[1, 6],
        xlim=(0.5, 6.5),
        ylim=(8e-6, 2e-2),
    )
    ax.minorticks_off()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.text(0.5, 0.79, "Frozen 3D field, one renewal", ha="center", va="bottom")
    save(fig, (ax,), args.output, "lattice_phase_rotation_error", args.format)

    manifest = {
        "width_cm": 12.5,
        "font_size_pt": theme.THESIS_FONT_SIZE_PT,
        "baseline_audit": str(args.results / "audit/audit.json"),
        "phase_control": str(args.results / "lattice-phase-production/phase.json"),
        "advancing_comparisons": [str(p) for p in args.comparisons],
        "latest_candidate_time": max(by_time, default=None),
        "note": "Reference reattachment uses FVM; baseline and candidate reattachment use their VPM profiles beyond the small FVM domain. Lines join sampled states only.",
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, nargs="*", default=[])
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()
