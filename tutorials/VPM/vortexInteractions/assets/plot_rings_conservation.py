#!/usr/bin/env python3
"""Conservation & structure-destruction audit — ``rings_conservation.png``.

Two stacked panels for every case discovered under ``solution/`` (read from the
solver log):

  (top)    Particle-count growth N(t)/N₀ — a flat line is a fixed budget; a ramp
           into a plateau is the remeshing hitting ``max_particles`` and being
           forced to discard vorticity.
  (bottom) Linear-impulse drift |I(t)−I₀|/|I₀| — the physically conserved
           invariant for an unbounded vortex flow.  A stabilizer that "works" by
           throwing away circulation shows up here and in the printed
           destroyed-circulation column.

Color encodes the stabilization method, linestyle the interaction family — the
same key shared by every comparison figure (see ``_common.case_style``).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASSETS_DIR))
from _common import (  # noqa: E402
    T_REF,
    build_arg_parser,
    case_style,
    compact_case_legend_handles,
    compounded_discarded_fraction,
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
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    load_theme()
    fig, (ax_n, ax_imp) = plt.subplots(2, 1, figsize=figure_size("wide_stacked"), sharex=True)

    plotted = False
    discard_rows: list[tuple[str, float]] = []
    for case_dir in discover_cases(args.solution_dir):
        df = read_integrals(case_dir)
        if df is None or len(df) == 0:
            continue
        st = case_style(case_dir.name)
        t_star = df["time"].to_numpy(float) / T_REF
        common = dict(
            color=st["color"],
            linestyle=st["linestyle"],
            lw=st["linewidth"],
            marker=st["marker"],
            ms=st["markersize"],
            markevery=mark_every("energy"),
            mew=st["markeredgewidth"],
        )

        if "n_particles" in df.columns:
            n = df["n_particles"].to_numpy(float)
            if np.isfinite(n[0]) and n[0] > 0:
                ax_n.plot(t_star, n / n[0], label=st["label"], **common)
                plotted = True

        imp_cols = [f"impulse_{axis}" for axis in "xyz"]
        if all(col in df.columns for col in imp_cols):
            imp = df[imp_cols].to_numpy(float)
            imp0 = float(np.linalg.norm(imp[0]))
            if imp0 > 1e-30:
                drift = np.linalg.norm(imp - imp[0], axis=1) / imp0
                ax_imp.plot(t_star, np.maximum(drift, 1e-9), **common)

        discard_rows.append((case_dir.name, compounded_discarded_fraction(case_dir)))

    ax_n.set_ylabel(r"Particle count, $N/N_0$")
    ax_n.set_title("Structure destruction — particle growth and impulse drift")
    ax_imp.set_yscale("log")
    ax_imp.set_xlabel(r"Normalized time, $t\,\Gamma_0 / R_0^2$")
    ax_imp.set_ylabel(r"Impulse drift, $|I-I_0|/|I_0|$")

    if plotted:
        fig.legend(
            handles=compact_case_legend_handles(),
            ncol=5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
        )

    save_fig(
        fig,
        figs / "rings_conservation.png",
        dpi=args.dpi,
        figure_format=args.format,
        tight_rect=(0.0, 0.16, 1.0, 1.0),
    )

    if discard_rows:
        print("\n=== Circulation destroyed by capped/thresholded remeshing ===")
        for name, frac in sorted(discard_rows, key=lambda r: r[1], reverse=True):
            if frac > 0.0:
                print(f"  {name:<22} {100 * frac:6.2f}%")
        if not any(frac > 0.0 for _, frac in discard_rows):
            print("  (none — no run discarded circulation)")


if __name__ == "__main__":
    main()
