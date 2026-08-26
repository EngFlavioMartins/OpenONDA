#!/usr/bin/env python3
"""Co-rotating vortex merger: angle, core radius, and vortex_separation histories.

Reads the field-based vortex diagnostics (``field_diagnostics.csv``, from the
z=L/4 velocity/vorticity plane) for each viscous scheme.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .vortex_diagnostics import (
        REF_DIR,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        unwrap_pair_orientation,
    )
else:
    from vortex_diagnostics import (
        REF_DIR,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        figure_size,
        load_theme,
        resolve_runtime_physics,
        save_fig,
        unwrap_pair_orientation,
    )


THETA_REF = REF_DIR / "theta_vs_tau.csv"
A2_REF = REF_DIR / "a2_over_b02.csv"
B_DIMENSIONAL_REF = REF_DIR / "b_over_b0_time.csv"


def extract_merging_timeseries(
    samples_dir: Path,
    scheme: str,
    kinematic_viscosity: float,
    vortex_separation: float,
    core_radius: float,
) -> dict | None:
    path = samples_dir / f"merging_{scheme}" / "field_diagnostics.csv"
    if not path.is_file():
        return None
    try:
        data = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["time", "step"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [merging] skipping unreadable live CSV for {scheme!r}: {exc}")
        return None
    data = data.sort_values("step").drop_duplicates("step", keep="last")
    if data.empty:
        return None

    angle = data["angle_radians"].to_numpy(float)
    finite = np.isfinite(angle)
    angle_degrees = np.full_like(angle, np.nan)
    if finite.any():
        # The axis joining identical vortices is undirected: swapping center
        # labels changes phi by pi but must not change the physical angle.
        unwrapped = unwrap_pair_orientation(angle[finite])
        angle_degrees[finite] = np.degrees(unwrapped - unwrapped[0])

    time = data["time"].to_numpy(float)
    return {
        "tau": kinematic_viscosity * time / core_radius**2,
        "theta_deg": angle_degrees,
        "a_c2_over_b02": data["mean_core_radius"].to_numpy(float) ** 2 / vortex_separation**2,
        "b_over_b0": data["vortex_separation"].to_numpy(float) / vortex_separation,
    }


def plot_merging_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"merging_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(
        samples_dir,
        args.circulation,
        args.kinematic_viscosity,
        args.b0,
        args.a0_over_b0,
        prefix="merging",
    )
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    a0 = runtime["velocity_peak_radius0"]
    b0 = runtime["vortex_separation"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    fig.subplots_adjust(hspace=0.09, top=0.95, bottom=0.23, left=0.10, right=0.98)

    plotted_schemes = []
    for scheme in SCHEMES:
        timeseries = extract_merging_timeseries(
            samples_dir,
            scheme,
            run_kinematic_viscosity,
            b0,
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

    reference_options = {
        "color": colors["reference"],
        "linestyle": "-",
        "linewidth": 1.0,
        "zorder": 100,
        "label": r"Cerretelli \& Williamson (2003)",
    }
    scale = (a0 / b0) ** 2

    for axis, path in ((axes[0], THETA_REF), (axes[1], A2_REF)):
        if path.exists():
            reference = np.loadtxt(path, delimiter=",")
            axis.plot(reference[:, 0] / scale, reference[:, 1], **reference_options)
    if B_DIMENSIONAL_REF.exists():
        print(
            "  [merging] vortex_separation literature curve retained as dimensional source data "
            "but not overlaid: experimental kinematic_viscosity/b0^2 provenance is not yet recorded"
        )

    axes[0].set_ylabel(r"$\theta$ [deg]")
    axes[0].set_title(r"Merging vortex characteristics")
    axes[0].set_ylim([-20, 900])

    axes[1].set_ylabel(r"$a_c^2 / b_0^2$")
    axes[1].set_ylim([-0.1, 0.70])

    axes[2].set_xlabel(r"$\nu t / a_{c,0}^2$")
    axes[2].set_ylabel(r"$b / b_0$")
    axes[2].set_xlim([0, 3.2])
    axes[2].set_ylim([0, 2.0])

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    parser = build_arg_parser("Co-rotating vortex merger diagnostics comparison.")
    return plot_merging_case(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
