#!/usr/bin/env python3
"""Energy and enstrophy evolution — ``rings_energy.png``.

Two stacked panels for every case discovered under ``solution/`` (read from
the VPM flow-integral sampler):

  (top)    Kinetic energy   E(t)/E(0)
  (bottom) Enstrophy        ε(t)/ε(0)

Color encodes the numerical method, linestyle the interaction family - the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
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
    args = build_arg_parser("Kinetic energy E/E₀ and enstrophy ε/ε₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, (ax_e, ax_w) = plt.subplots(2, 1, figsize=figure_size("wide_stacked"), sharex=True)

    plotted = False
    for case_dir in discover_cases(args.solution_dir):
        df = read_integrals(case_dir)
        if df is None or "total_kinetic_energy" not in df.columns or len(df) == 0:
            continue
        st = case_style(case_dir.name)
        nondimensional_time = df["time"].to_numpy(float) / REFERENCE_TIME
        ke = df["total_kinetic_energy"].to_numpy(float)
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every("total_kinetic_energy"),
            mew=st["markeredgewidth"],
        )
        if ke[0] > 0.0:
            ax_e.plot(nondimensional_time, ke / ke[0], label=st["label"], **common)
            plotted = True
        if "total_enstrophy" in df.columns:
            ens = df["total_enstrophy"].to_numpy(float)
            if ens[0] > 0.0:
                ax_w.plot(nondimensional_time, ens / ens[0], **common)

    ax_e.set_ylabel(r"Kinetic energy, $E / E_0$")
    ax_e.set_title("Energy and enstrophy evolution")
    ax_w.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_w.set_ylabel(r"Enstrophy, $\varepsilon / \varepsilon_0$")
    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        figs / "rings_energy.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.16, 1.0, 1.0),
    )


if __name__ == "__main__":
    main()
