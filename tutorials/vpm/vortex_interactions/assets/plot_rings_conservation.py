#!/usr/bin/env python3
"""Particle count and all conserved vector moments — ``rings_conservation.png``.

Four stacked panels, one per family-independent quantity, from the VPM
flow-integral sampler:

  (top)     Particle count     N/N₀
  (2nd)     Net vortex strength |ΣΓ - ΣΓ₀|/Γ₀
  (3rd)     Linear impulse     |I - I₀|/(Γ₀R₀²)
  (bottom)  Angular impulse    |A - A₀|/(Γ₀R₀³)

For a closed system of vortex rings the momentum analogues — net vortex
strength, linear impulse and finite-core angular impulse — are conserved by
the inviscid dynamics, so their drift is a direct measure of numerical
(dish)honesty. Closed rings can carry a nearly zero net vector moment, so each
drift is scaled by a physical ring scale (Γ₀, Γ₀R₀², Γ₀R₀³) rather than by a
cancellation-prone initial norm. Particle count is included because it exposes
the difference between the fixed-cloud baselines (constant N) and the
stabilized LES, which splits filaments and grows N.

Every case carries its own colour, marker and legend entry — the same key
shared by every comparison figure (see ``ring_metrics.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ring_metrics import (
    RING_CIRCULATION,
    RING_RADIUS,
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    case_legend_handles,
    discover_cases,
    figure_size,
    load_theme,
    mark_every,
    read_integrals,
    save_fig,
)


def main() -> None:
    args = build_arg_parser(
        "Particle-count growth and impulse drift (structure-destruction audit)."
    ).parse_args()

    load_theme()
    fig, axes = plt.subplots(4, 1, figsize=figure_size("stacked"), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.19, hspace=0.12)
    ax_n, ax_circ, ax_imp, ax_ang = axes

    plotted: list[str] = []
    for case_dir in discover_cases(args.solution_dir):
        df = read_integrals(case_dir)
        if df is None or len(df) == 0:
            continue
        st = case_style(case_dir.name)
        nondimensional_time = df["time"].to_numpy(float) / REFERENCE_TIME
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every("total_kinetic_energy"),
            mew=st["markeredgewidth"],
        )

        if "n_particles_total" in df.columns:
            n = df["n_particles_total"].to_numpy(float)
            if np.isfinite(n[0]) and n[0] > 0:
                ax_n.plot(nondimensional_time, n / n[0], label=st["label"], **common)
                plotted.append(case_dir.name)

        imp_cols = [f"linear_impulse_{axis}" for axis in "xyz"]
        strength_cols = [f"net_vortex_strength_{axis}" for axis in "xyz"]
        if all(col in df.columns for col in strength_cols):
            net_vortex_strength = df[strength_cols].to_numpy(float)
            drift = (
                np.linalg.norm(net_vortex_strength - net_vortex_strength[0], axis=1)
                / RING_CIRCULATION
            )
            ax_circ.plot(nondimensional_time, np.maximum(drift, 1e-12), **common)

        if all(col in df.columns for col in imp_cols):
            imp = df[imp_cols].to_numpy(float)
            scale = max(float(np.linalg.norm(imp[0])), RING_CIRCULATION * RING_RADIUS**2)
            drift = np.linalg.norm(imp - imp[0], axis=1) / scale
            ax_imp.plot(nondimensional_time, np.maximum(drift, 1e-12), **common)

        angular_cols = [f"angular_impulse_{axis}" for axis in "xyz"]
        if all(col in df.columns for col in angular_cols):
            angular = df[angular_cols].to_numpy(float)
            scale = max(float(np.linalg.norm(angular[0])), RING_CIRCULATION * RING_RADIUS**3)
            drift = np.linalg.norm(angular - angular[0], axis=1) / scale
            ax_ang.plot(nondimensional_time, np.maximum(drift, 1e-12), **common)

    ax_n.set_ylabel(r"$N/N_0$")
    ax_n.set_title("Conservation")
    for axis in (ax_circ, ax_imp, ax_ang):
        axis.set_yscale("log")
    ax_circ.set_ylabel(r"$|\Delta\sum\Gamma|/\Gamma_0$")
    ax_imp.set_yscale("log")
    ax_imp.set_ylabel(r"$|\Delta I|/(\Gamma_0R_0^2)$")
    ax_ang.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_ang.set_ylabel(r"$|\Delta A|/(\Gamma_0R_0^3)$")

    if plotted:
        fig.legend(
            handles=case_legend_handles(plotted),
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )

    save_fig(
        fig,
        Path("figures") / "rings_conservation.png",
        dpi=args.dpi,
        figure_format=args.format,
    )


if __name__ == "__main__":
    main()
