#!/usr/bin/env python3
"""Counter-rotating vortex dipole — core trajectory and radius comparison.

Reads the field-based vortex diagnostics (``field_diagnostics.csv``, from the
z=L/4 velocity/vorticity plane) for each viscous scheme and plots:
  - core x-position  xc / a_{c,0}  vs  ν t / a_{c,0}²
  - core radius       a_c / a_{c,0}  vs  ν t / a_{c,0}²
  - 80%-region centre separation b / b_0
  - 50%-region vorticity-weighted aspect ratio E_50

The trajectory panel also includes the analytical translation of the finite
Lamb--Oseen vortex filaments.  The reference uses the fixed initial spacing
between the two filaments and the plane's finite-filament endpoint factor.

Saves: figures/dipole_comparison.png
"""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .postprocess import (
    SCHEME_DRAW_ORDER,
    SCHEMES,
    build_arg_parser,
    build_style_map,
    centered_subplots_adjust,
    connected_core_aspect,
    diffusion_only_dipole_features,
    load_dipole_feature_frame,
    load_theme,
    read_surface_field,
    resolve_runtime_physics,
    save_fig,
    scheme_zorder,
    theoretical_dipole_trajectory,
    validate_thesis_figure,
)

# Plot


def plot_dipole_case(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"dipole_comparison.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_physics(
        samples_dir,
        args.circulation,
        args.kinematic_viscosity,
        args.b0,
        args.a0_over_b0,
        prefix="dipole",
    )
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    a0 = runtime["velocity_peak_radius0"]
    colors, theme = load_theme()
    style_map = build_style_map(colors)

    frames = {
        scheme: frame
        for scheme in SCHEMES
        if (frame := load_dipole_feature_frame(samples_dir, scheme)) is not None
    }
    if not frames:
        out.unlink(missing_ok=True)
        print("  [dipole] no sampled trajectories; figure not generated")
        return 0

    common_steps = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    common_steps = [
        step
        for step in common_steps
        if all(
            (samples_dir / f"dipole_{scheme}" / f"dipole_{scheme}_zq_{step:06d}.vts").is_file()
            for scheme in frames
        )
    ]
    if not common_steps:
        out.unlink(missing_ok=True)
        print("  [dipole] no common sampled fields; figure not generated")
        return 0

    reference_scheme = next(iter(frames))
    time = frames[reference_scheme].loc[common_steps, "time"].to_numpy(float)
    tau = run_kinematic_viscosity * time / (a0**2)
    aspects: dict[str, np.ndarray] = {}
    template = None
    for scheme, frame in frames.items():
        values = []
        for step in common_steps:
            field = read_surface_field(
                samples_dir / f"dipole_{scheme}" / f"dipole_{scheme}_zq_{step:06d}.vts"
            )
            if template is None:
                template = field
            values.append(connected_core_aspect(field))
        aspects[scheme] = np.asarray(values, dtype=float)
    if template is None:
        raise RuntimeError("no dipole surface field was readable")

    diffusion_separation, diffusion_aspect = diffusion_only_dipole_features(
        template,
        time,
        runtime["circulation"],
        runtime["ac0"],
        run_kinematic_viscosity,
        runtime["vortex_separation"],
    )

    fig, axes_grid = plt.subplots(2, 2, figsize=(125 / 25.4, 95 / 25.4), sharex=True)
    centered_subplots_adjust(fig, outer=0.13, bottom=0.30, top=0.93, wspace=0.42, hspace=0.42)
    axes = axes_grid.ravel()

    plotted_schemes = []
    for scheme in SCHEME_DRAW_ORDER:
        if scheme not in frames:
            continue
        frame = frames[scheme].loc[common_steps]
        st = style_map[scheme]
        plot_kw = {
            "color": st["color"],
            "label": st["label"],
            "marker": st["marker"],
            "markersize": 2.0,
            "linestyle": "None",
            "linewidth": 1.0,
            "zorder": 150 if scheme == "rwm" else scheme_zorder(scheme),
        }
        series = (
            (axes[0], "vortex_centre_0_x", a0),
            (axes[1], "core_radius_0", a0),
            (axes[2], "vortex_separation", runtime["vortex_separation"]),
        )
        for axis, column, scale in series:
            values = frame[column].to_numpy(float) / scale
            finite = np.isfinite(values)
            axis.plot(tau[finite], values[finite], **plot_kw)
            if scheme == "rwm":
                lower_name = f"{column}_ci_lower"
                upper_name = f"{column}_ci_upper"
                if lower_name in frame and upper_name in frame:
                    lower = frame[lower_name].to_numpy(float) / scale
                    upper = frame[upper_name].to_numpy(float) / scale
                    interval = finite & np.isfinite(lower) & np.isfinite(upper)
                    axis.fill_between(
                        tau[interval],
                        lower[interval],
                        upper[interval],
                        color=st["color"],
                        alpha=0.18,
                        linewidth=0,
                        zorder=scheme_zorder(scheme) - 1,
                    )
        aspect = aspects[scheme]
        finite_aspect = np.isfinite(aspect)
        axes[3].plot(tau[finite_aspect], aspect[finite_aspect], **plot_kw)
        plotted_schemes.append(scheme)

    print(f"  [dipole] plotting {len(plotted_schemes)}/{len(SCHEMES)} methods")

    # The pair is initialized as two finite columns.  Evaluate the fixed-core
    # reference at the sampled times and include the exact initial condition.
    time_ref = np.concatenate(([0.0], time))
    tau_ref = run_kinematic_viscosity * time_ref / (a0**2)
    x_ref = theoretical_dipole_trajectory(
        time_ref,
        runtime["circulation"],
        runtime["vortex_separation"],
        run_kinematic_viscosity,
        runtime["t0"],
        runtime["column_length"],
    )
    reference_options = dict(theme.REFERENCE_STYLE)
    reference_options.update(label="Fixed circular-core reference", zorder=100)
    axes[0].plot(tau_ref, x_ref / a0, **reference_options)

    surrogate_options = {
        "color": colors.get("GREY_DARK", "0.35"),
        "linestyle": ":",
        "linewidth": 1.0,
        "label": "Diffusion-only pair",
        "zorder": 90,
    }
    axes[2].plot(
        tau,
        diffusion_separation / runtime["vortex_separation"],
        **surrogate_options,
    )
    axes[3].plot(tau, diffusion_aspect, **surrogate_options)

    labels = (
        ("Core trajectory", r"$x_c/a_{c,0}$"),
        ("Core radius", r"$a_c/a_{c,0}$"),
        ("Core separation", r"$b/b_0$"),
        ("High-vorticity aspect", r"$E_{50}$"),
    )
    for axis, (title, ylabel) in zip(axes, labels, strict=True):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    for axis in axes[:2]:
        axis.tick_params(labelbottom=False)
    fig.supxlabel(r"$\nu t/a_{c,0}^2$", y=0.205)

    handles, legend_labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.004),
            borderpad=0.25,
            labelspacing=0.30,
            handletextpad=0.55,
            columnspacing=1.20,
        )
    validate_thesis_figure(fig, axes)
    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Counter-rotating dipole trajectory and core-radius comparison.")
    return plot_dipole_case(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
