#!/usr/bin/env python3
"""Energy dissipation — normalised power across all three test cases.

Reads flow-integral CSV files (``samples/flow_integrals.csv``) exported by
the VPM solver when logging is active.

Plots one column for each physics case: single vortex, vortex dipole, and
co-rotating merger. The upper row shows total kinetic energy and the lower row
shows its rate.

Continuous lines show the enstrophy-based sink -2νZ. Sparse filled circles
show dE/dt from direct/Fourier energy differences, including the explicitly
labelled finite transition estimate when the scalable diagnostic first takes
over. Colours are consistent per scheme across all three panels.

Saves: figures/lamboseen_energy.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __package__:
    from .postprocess import (
        ENERGY_CASES,
        SCHEME_DRAW_ORDER,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        figure_size,
        load_theme,
        prepend_initial_point,
        read_flow_integrals,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
    )
else:
    from postprocess import (
        ENERGY_CASES,
        SCHEME_DRAW_ORDER,
        SCHEMES,
        build_arg_parser,
        build_style_map,
        figure_size,
        load_theme,
        prepend_initial_point,
        read_flow_integrals,
        resolve_runtime_physics,
        save_fig,
        scheme_zorder,
    )


# =============================================================
# Plot
# =============================================================


def plot_case_panel(
    energy_ax,
    rate_ax,
    samples_dir: Path,
    case_prefix: str,
    title: str,
    n_vortices: int,
    style_map: dict,
    tau_scale: float,
    energy_ref: float,
    p_ref: float,
    circulation: float,
    t0: float,
    column_length: float,
) -> float:
    energy_ax.set_title(title)
    latest_tau = 0.0
    for scheme in SCHEME_DRAW_ORDER:
        csv_path = samples_dir / f"{case_prefix}_{scheme}" / "flow_integrals.csv"
        data = read_flow_integrals(csv_path)
        if data is None or "total_kinetic_energy" not in data:
            continue
        data = prepend_initial_point(data, circulation, t0, n_vortices, column_length)
        st = style_map[scheme]
        tau = data["time"] * tau_scale
        energy = data["total_kinetic_energy"] / (n_vortices * energy_ref)
        energy_rate = data["kinetic_energy_rate"] / p_ref
        enstrophy_rate = data["viscous_kinetic_energy_rate"] / p_ref

        energy_ax.plot(
            tau,
            energy,
            color=st["color"],
            linestyle="-",
            linewidth=1.0,
            alpha=0.90,
            zorder=scheme_zorder(scheme),
        )
        rate_ax.plot(
            tau,
            enstrophy_rate,
            color=st["color"],
            linestyle="-",
            linewidth=1.0,
            alpha=0.85,
            zorder=scheme_zorder(scheme),
        )
        if scheme == "rwm":
            if "total_kinetic_energy_ci_lower" in data and "total_kinetic_energy_ci_upper" in data:
                lower = data["total_kinetic_energy_ci_lower"] / (n_vortices * energy_ref)
                upper = data["total_kinetic_energy_ci_upper"] / (n_vortices * energy_ref)
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                energy_ax.fill_between(
                    tau[finite_interval],
                    lower[finite_interval],
                    upper[finite_interval],
                    color=st["color"],
                    alpha=0.10,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )
            for measure in ("kinetic_energy_rate", "viscous_kinetic_energy_rate"):
                lower_key = f"{measure}_ci_lower"
                upper_key = f"{measure}_ci_upper"
                if lower_key not in data or upper_key not in data:
                    continue
                lower = data[lower_key] / p_ref
                upper = data[upper_key] / p_ref
                finite_interval = np.isfinite(lower) & np.isfinite(upper)
                rate_ax.fill_between(
                    tau[finite_interval],
                    lower[finite_interval],
                    upper[finite_interval],
                    color=st["color"],
                    alpha=0.10,
                    linewidth=0,
                    zorder=scheme_zorder(scheme) - 1,
                )

        finite_energy = np.flatnonzero(np.isfinite(energy_rate))
        marker_stride = max(1, len(finite_energy) // 12)
        marker_indices = finite_energy[::marker_stride]
        if finite_energy.size and marker_indices[-1] != finite_energy[-1]:
            marker_indices = np.append(marker_indices, finite_energy[-1])
        rate_ax.plot(
            tau[marker_indices],
            energy_rate[marker_indices],
            color=st["color"],
            marker="o",
            markersize=2.6,
            linestyle="None",
            alpha=0.90,
            zorder=scheme_zorder(scheme, offset=1),
        )
        latest_tau = max(latest_tau, float(tau.max()))
    return latest_tau


def plot_energy_enstrophy(args) -> int:
    samples_dir = Path(args.samples_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"lamboseen_energy.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, theme = load_theme()
    if theme is not None and hasattr(theme, "set_style"):
        theme.set_style()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(
        samples_dir, args.circulation, args.kinematic_viscosity, args.b0, args.a0_over_b0
    )
    run_kinematic_viscosity = runtime["kinematic_viscosity"]
    a0 = runtime["velocity_peak_radius0"]
    run_t0 = runtime["t0"]
    run_circulation = runtime["circulation"]
    column_length = runtime["column_length"]

    tau_scale = run_kinematic_viscosity / (a0**2)
    # flow_integrals.csv contains 3-D totals.  The Lamb-Oseen dissipation
    # formula and its natural scale are per unit length, so both must carry L.
    p_ref = run_kinematic_viscosity * run_circulation**2 * column_length / (a0**2)
    energy_ref = run_circulation**2 * column_length
    base_width, base_height = figure_size("trajectory")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(base_width, 1.75 * base_height),
        sharex="col",
        sharey="row",
    )
    fig.subplots_adjust(wspace=0.09, hspace=0.12, top=0.94, bottom=0.20, left=0.14, right=0.86)

    plotted = False
    for column, (case_prefix, title, n_vortices) in enumerate(ENERGY_CASES):
        latest_tau = plot_case_panel(
            axes[0, column],
            axes[1, column],
            samples_dir,
            case_prefix,
            title,
            n_vortices,
            style_map,
            tau_scale,
            energy_ref,
            p_ref,
            run_circulation,
            run_t0,
            column_length,
        )
        plotted |= latest_tau > 0.0

        axes[1, column].set_xlabel(r"$\nu t / a_{c,0}^2$")

    if not plotted:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [energy] no sampled flow integrals; figure not generated")
        return 0

    axes[0, 0].set_ylabel(r"$E / (N_v\Gamma^2 L)$")
    axes[1, 0].set_ylabel(r"$(dE/dt) / (\nu\Gamma^2 L / a_{c,0}^2)$")
    axes[1, 0].set_ylim([-5e-1, -5e-3])

    # sharey="row" links the three rate panels.
    for ax in axes[1, :]:
        ax.set_yscale("symlog", linthresh=0.01)
    for ax in axes[1, :]:
        ax.axhspan(0.0, 1e-1, color=colors["background_light"], linewidth=0, zorder=0)

    available_schemes = [
        scheme
        for scheme in SCHEMES
        if any(
            read_flow_integrals(samples_dir / f"{case_prefix}_{scheme}" / "flow_integrals.csv")
            is not None
            for case_prefix, _, _ in ENERGY_CASES
        )
    ]
    handles: list = []
    for scheme in available_schemes:
        st = style_map[scheme]
        handles.append(
            Line2D(
                [0],
                [0],
                color=st["color"],
                linestyle="-",
                marker="None",
                linewidth=1.0,
                label=st["label"],
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color=colors["reference"],
            linestyle="None",
            marker="o",
            markersize=4,
            mfc=colors["reference"],
            label=r"$dE/dt$",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=colors["reference"],
            linestyle="-",
            marker="None",
            linewidth=1.0,
            label=r"$-2\nu Z$",
        )
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.00),
    )

    save_fig(fig, out, args.dpi)
    return 0


def main() -> int:
    p = build_arg_parser("Energy balance: dE/dt")
    return plot_energy_enstrophy(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
