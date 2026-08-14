#!/usr/bin/env python3
"""Co-rotating vortex merger: angle, core area, and separation histories."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from ._common import (
        BETA_RMAX,
        REF_DIR,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        load_theme,
        figure_size,
        resolve_runtime_physics,
        save_fig,
    )
else:
    from _common import (
        BETA_RMAX,
        REF_DIR,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        load_theme,
        figure_size,
        resolve_runtime_physics,
        save_fig,
    )


THETA_REF = REF_DIR / "theta_vs_tau.csv"
A2_REF = REF_DIR / "a2_over_b02.csv"
B_REF = REF_DIR / "b_over_b0_tau.csv"


def extract_merging_timeseries(
    samples_dir: Path,
    scheme: str,
    viscosity: float,
    separation: float,
    core_radius: float,
) -> dict | None:
    path = samples_dir / f"merging_{scheme}" / "pair_diagnostics.csv"
    if not path.is_file():
        return None
    data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["flow_time", "time_step"])
    data = data.sort_values("time_step").drop_duplicates("time_step", keep="last")
    if data.empty:
        return None

    merged = data["merged"].astype(str).str.lower().eq("true").to_numpy()
    pair_separation = data["separation"].to_numpy(float)
    collapsed = merged & (np.isnan(pair_separation) | (pair_separation < 0.5 * separation))
    if collapsed.any():
        data = data.iloc[: int(np.flatnonzero(collapsed)[0])]
    if data.empty:
        return None

    circulation = data["surface_circulation"].to_numpy(float)
    drifted = np.abs(circulation / circulation[0] - 1.0) > 0.50
    if drifted.any():
        data = data.iloc[: int(np.flatnonzero(drifted)[0])]
    if data.empty:
        return None

    angle = data["angle_rad"].to_numpy(float)
    finite = np.isfinite(angle)
    angle_degrees = np.full_like(angle, np.nan)
    if finite.any():
        unwrapped = np.unwrap(angle[finite])
        angle_degrees[finite] = np.degrees(unwrapped - unwrapped[0])

    time = data["flow_time"].to_numpy(float)
    return {
        "tau": viscosity * time / core_radius**2,
        "theta_deg": angle_degrees,
        "a_c2_over_b02": (2.0 * BETA_RMAX**2 * data["core_area"].to_numpy(float) / separation**2),
        "b_over_b0": data["separation"].to_numpy(float) / separation,
    }


def plot_merging_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"merging_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(samples_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    a0 = runtime["ac0"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    fig.subplots_adjust(hspace=0.09, top=0.95, bottom=0.23, left=0.10, right=0.98)

    plotted_schemes = []
    for scheme in SCHEMES:
        timeseries = extract_merging_timeseries(
            samples_dir,
            scheme,
            run_nu,
            args.b0,
            a0,
        )
        if timeseries is None:
            print(f"  [merging] skipping {scheme!r} — no data")
            continue
        style = style_map[scheme]
        plot_options = {
            "color": style["color"],
            "label": style["label"],
            "marker": style["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
        }
        axes[0].plot(timeseries["tau"], timeseries["theta_deg"], **plot_options)
        axes[1].plot(timeseries["tau"], timeseries["a_c2_over_b02"], **plot_options)
        axes[2].plot(timeseries["tau"], timeseries["b_over_b0"], **plot_options)
        plotted_schemes.append(scheme)

    if not plotted_schemes:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [merging] no sampled diagnostics; figure not generated")
        return 0

    print(f"  [merging] plotting {len(plotted_schemes)}/{len(SCHEMES)} methods")

    reference_options = {
        "color": colors["reference"],
        "linestyle": "-",
        "linewidth": 1.0,
        "zorder": 100,
        "label": r"Cerretelli \& Williamson (2003)",
    }
    scale = args.a0_over_b0**2
    missing = [path.name for path in (THETA_REF, A2_REF, B_REF) if not path.exists()]
    if missing:
        print(
            f"  [merging] WARNING: no reference curve for {', '.join(missing)}; "
            f"expected under {REF_DIR}"
        )
    for axis, path in ((axes[0], THETA_REF), (axes[1], A2_REF), (axes[2], B_REF)):
        if path.exists():
            reference = np.loadtxt(path, delimiter=",")
            axis.plot(reference[:, 0] / scale, reference[:, 1], **reference_options)

    axes[0].set_ylabel(r"$\theta$ [deg]")
    axes[0].set_title(r"Merging vortex characteristics")
    axes[0].set_ylim([-20, 900])

    axes[1].set_ylabel(r"$a_c^2 / b_0^2$")
    axes[1].set_ylim([-0.1, 0.70])

    axes[2].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[2].set_ylabel(r"$b / b_0$")
    axes[2].set_xlim([0, 3.2])
    axes[2].set_ylim([0, 1.5])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    parser = build_arg_parser("Co-rotating vortex merger diagnostics comparison.")
    return plot_merging_case(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
