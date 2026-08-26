#!/usr/bin/env python3
"""Particle count and all conserved vector moments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    RING_CIRCULATION,
    RING_RADIUS,
    REFERENCE_TIME,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
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
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)
    ax_n, ax_circ, ax_imp, ax_ang = axes

    plotted = False
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
                plotted = True

        imp_cols = [f"impulse_{axis}" for axis in "xyz"]
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

    ax_n.set_ylabel(r"Particle count, $N/N_0$")
    ax_n.set_title("Conservation contract")
    for axis in (ax_circ, ax_imp, ax_ang):
        axis.set_yscale("log")
    ax_circ.set_ylabel(r"$|\sum\Gamma-\sum\Gamma_0|/\Gamma_0$")
    ax_imp.set_yscale("log")
    ax_imp.set_ylabel(r"$|I-I_0|/(\Gamma_0R_0^2)$")
    ax_ang.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_ang.set_ylabel(r"$|A-A_0|/(\Gamma_0R_0^3)$")

    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        Path("figures") / "rings_conservation.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.10, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
