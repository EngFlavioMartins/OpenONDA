#!/usr/bin/env python3
"""Wall-normal velocity profiles in similarity variables vs the Blasius
solution.  Profiles from every station must collapse onto the single curve
u/U = f'(eta) if the solver reproduces the laminar boundary layer."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    COLORS,
    FREESTREAM_SPEED,
    blasius_solution,
    build_arg_parser,
    figure_size,
    load_csv_columns,
    save_fig,
)

STATION_MARKERS = {0.25: "o", 0.5: "s", 0.75: "^"}


def main():
    args = build_arg_parser().parse_args()
    kinematic_viscosity = FREESTREAM_SPEED * 1.0 / args.Re
    data = load_csv_columns(Path(args.solution_dir) / "profiles.csv")
    if not data:
        return

    eta_ref, fprime_ref = blasius_solution()

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(
        eta_ref, fprime_ref, color=COLORS["reference"], linewidth=1.4, label="Blasius $f'(\\eta)$"
    )

    max_err = 0.0
    for station in sorted(set(data["station"])):
        sel = data["station"] == station
        y, u = data["position_y"][sel], data["velocity_x"][sel]
        eta = y * np.sqrt(FREESTREAM_SPEED / (kinematic_viscosity * station))
        u_norm = u / FREESTREAM_SPEED
        marker = STATION_MARKERS.get(station, "d")
        ax.plot(
            eta, u_norm, marker, markersize=3.5, linestyle="none", label=f"FVM $x/L$ = {station:g}"
        )
        # Error against Blasius inside the layer (eta <= 6).
        inside = eta <= 6.0
        u_ref = np.interp(eta[inside], eta_ref, fprime_ref)
        err = float(np.max(np.abs(u_norm[inside] - u_ref))) if inside.any() else 0.0
        max_err = max(max_err, err)
        print(f"  x/L = {station:g}: max |u/U - f'(eta)| = {err:.4f} (eta <= 6)")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"$\eta = y \sqrt{U_\infty / (\nu x)}$")
    ax.set_ylabel(r"$u / U_\infty$")
    ax.set_title(f"Flat-plate profiles vs Blasius (Re$_L$ = {args.Re:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, "blasius_profiles.png", args.figures_dir, dpi=args.dpi, figure_format=args.format)
    print(
        f"  overall max profile error: {max_err:.4f}"
        f"  [{'OK' if max_err < 0.05 else 'OUT OF BAND'} — target < 0.05]"
    )


if __name__ == "__main__":
    main()
