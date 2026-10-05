"""Co-rotating vortex merger: angle, core radius, and vortex_separation histories.

Reads the field-based vortex diagnostics (``field_diagnostics.csv``, from the
z=L/4 velocity/vorticity plane) for each viscous scheme.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from .postprocess import (
    MERGING_NORMALIZED_END_TIME,
    SCHEME_DRAW_ORDER,
    build_arg_parser,
    build_style_map,
    centered_subplots_adjust,
    extract_merging_timeseries,
    figure_size,
    load_merging_references,
    load_theme,
    resolve_runtime_physics,
    save_fig,
    scheme_zorder,
)


def plot_merging_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = args.format
    out = Path(args.figures_dir) / f"merging_comparison.{fmt}"
    colors, _ = load_theme()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(samples_dir, prefix="merging")
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    a0 = runtime["velocity_peak_radius0"]
    b0 = runtime["vortex_separation"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("stacked_tall"))
    centered_subplots_adjust(fig, outer=0.1215, hspace=0.09, top=0.953, bottom=0.22)
    plotted_schemes = []
    for scheme in SCHEME_DRAW_ORDER:
        timeseries = extract_merging_timeseries(
            samples_dir, scheme, run_kinematic_viscosity, b0, a0
        )
        style = style_map[scheme]
        plot_options = {
            "color": style["color"],
            "label": style["label"],
            "marker": style["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
            "zorder": scheme_zorder(scheme),
        }
        tau = timeseries["tau"]
        theta = timeseries["theta_deg"]
        core_size = timeseries["a_c2_over_b02"]
        separation = timeseries["b_over_b0"]
        angle_mask = np.isfinite(tau) & np.isfinite(theta)
        core_mask = np.isfinite(tau) & np.isfinite(core_size)
        separation_mask = np.isfinite(tau) & np.isfinite(separation)
        axes[0].plot(tau[angle_mask], theta[angle_mask], **plot_options)
        axes[1].plot(tau[core_mask], core_size[core_mask], **plot_options)
        axes[2].plot(tau[separation_mask], separation[separation_mask], **plot_options)
        if scheme == "rwm":
            for axis, mask, lower_key, upper_key in (
                (axes[0], angle_mask, "theta_ci_lower", "theta_ci_upper"),
                (axes[1], core_mask, "a_c2_over_b02_ci_lower", "a_c2_over_b02_ci_upper"),
                (axes[2], separation_mask, "b_over_b0_ci_lower", "b_over_b0_ci_upper"),
            ):
                lower = timeseries[lower_key]
                upper = timeseries[upper_key]
                finite_interval = mask & np.isfinite(lower) & np.isfinite(upper)
                axis.fill_between(
                    tau,
                    lower,
                    upper,
                    where=finite_interval,
                    color=style["color"],
                    alpha=0.18,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )
        plotted_schemes.append(scheme)
    reference_options = {
        "color": colors["reference"],
        "linestyle": "--",
        "linewidth": 1.0,
        "zorder": 100,
        "label": "Experiment",
    }
    references = load_merging_references(a0, b0)
    for axis, name in ((axes[0], "theta"), (axes[1], "core"), (axes[2], "separation")):
        if name in references:
            reference = references[name]
            axis.plot(reference[:, 0], reference[:, 1], **reference_options)
    axes[0].set_ylabel("$\\theta$ [deg]")
    axes[0].set_title("Merging vortex characteristics")
    axes[1].set_ylabel("$a_c^2 / b_0^2$")
    axes[2].set_xlabel("$\\nu t / a_{c,0}^2$")
    axes[2].set_ylabel("$b / b_0$")
    axes[2].set_xlim([0, MERGING_NORMALIZED_END_TIME + 0.1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    parser = build_arg_parser("Co-rotating vortex merger diagnostics comparison.")
    return plot_merging_case(parser.parse_args())


if __name__ == "__main__":
    main()
