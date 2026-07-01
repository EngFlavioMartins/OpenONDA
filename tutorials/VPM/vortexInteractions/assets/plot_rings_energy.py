#!/usr/bin/env python3
"""Energy and enstrophy evolution — ``rings_energy.png``.

Two stacked panels for every case discovered under ``solution/`` (read from
the solver log's ``FLOW DIAGNOSTICS`` sections):

  (top)    Kinetic energy   E(t)/E(0)
  (bottom) Enstrophy        ε(t)/ε(0)

Colour encodes the numerical variant, linestyle the physics family — the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    CM,
    T_REF,
    build_arg_parser,
    case_style,
    discover_cases,
    load_theme,
    read_integrals,
    save_fig,
)


def main() -> None:
    args = build_arg_parser("Kinetic energy E/E₀ and enstrophy ε/ε₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, (ax_e, ax_w) = plt.subplots(2, 1, figsize=(12.8 * CM, 11.0 * CM), sharex=True)

    plotted = False
    for case_dir in discover_cases(args.solution_dir):
        df = read_integrals(case_dir)
        if df is None or "kinetic_energy" not in df.columns or len(df) == 0:
            continue
        st = case_style(case_dir.name)
        t_star = df["time"].to_numpy(float) / T_REF
        ke = df["kinetic_energy"].to_numpy(float)
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=4,
            mew=0.4,
        )
        if ke[0] > 0.0:
            ax_e.plot(t_star, ke / ke[0], label=st["label"], **common)
            plotted = True
        if "enstrophy" in df.columns:
            ens = df["enstrophy"].to_numpy(float)
            if ens[0] > 0.0:
                ax_w.plot(t_star, ens / ens[0], **common)

    ax_e.set_ylabel(r"Kinetic energy, $E / E_0$")
    ax_e.set_title("Energy and enstrophy evolution")
    ax_w.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_w.set_ylabel(r"Enstrophy, $\varepsilon / \varepsilon_0$")
    if plotted:
        ax_e.legend(fontsize=10, ncol=2)

    save_fig(fig, figs / "rings_rings_energy.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
