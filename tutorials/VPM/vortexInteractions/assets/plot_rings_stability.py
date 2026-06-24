#!/usr/bin/env python3
"""Stability ladder — ``rings_stability.png``.

Tracks the peak circulation max|Γᵢ|/max|Γᵢ|₀ versus normalised time for every
case discovered under ``solution/`` (read from the per-step
``stability_metrics.csv``).  This is the blow-up indicator that orders the
stabilization ladder: a runaway curve marks the onset of numerical blow-up,
and where each curve ends marks that rung's survival time.  The dashed line is
the 50× blow-up threshold used by the solver to stop a diverging run.

Colour encodes the stabilization rung, linestyle the physics family — the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from _common import (
    CM,
    build_arg_parser,
    case_style,
    discover_cases,
    load_theme,
    read_metric,
    save_fig,
)

BLOWUP_FACTOR = 50.0  # mirrors rings_setup.py's max|Γ| > 50× initial stop


def main() -> None:
    args = build_arg_parser("Peak circulation max|Γᵢ|/max|Γᵢ|₀ vs t*.").parse_args()
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, ax = plt.subplots(figsize=(12.8 * CM, 7.0 * CM))

    plotted = False
    for case_dir in discover_cases(args.solution_dir):
        # Keep the full series (including the runaway) so survival time is visible.
        t_star, gmax = read_metric(case_dir, "max_gamma", truncate_blowup=False)
        if t_star.size == 0 or gmax[0] <= 0.0:
            continue
        st = case_style(case_dir.name)
        ax.plot(
            t_star,
            gmax / gmax[0],
            color=st["color"],
            linestyle=st["linestyle"],
            lw=1.1,
            marker=st["marker"],
            ms=3,
            markevery=20,
            mew=0.4,
            label=st["label"],
        )
        plotted = True

    ax.axhline(BLOWUP_FACTOR, color="0.4", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax.set_ylabel(r"Peak circulation, $\max_i|\Gamma_i| / \max_i|\Gamma_i|_0$")
    if plotted:
        ax.legend(fontsize=10, ncol=2, loc="upper left")

    save_fig(fig, figs / "rings_stability.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
