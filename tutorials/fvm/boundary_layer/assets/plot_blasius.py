#!/usr/bin/env python3
"""Wall-normal velocity profiles in similarity variables vs the Blasius
solution.  Profiles from every station must collapse onto the single curve
u/U = f'(eta) if the solver reproduces the laminar boundary layer."""

from pathlib import Path


import matplotlib.pyplot as plt
from openonda.plotting import centered_subplots_adjust
import numpy as np

from ._common import (  # noqa: E402
    COLORS,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    SOLUTION_DIR,
    blasius_solution,
    build_arg_parser,
    figure_size,
    read_csv_columns,
    save_fig,
)

STATION_MARKERS = {0.25: "o", 0.5: "s", 0.75: "^"}


def main():
    args = build_arg_parser().parse_args()
    kinematic_viscosity = FREESTREAM_SPEED * 1.0 / args.Re
    data = read_csv_columns(Path(SOLUTION_DIR) / "profiles.csv")

    eta_ref, fprime_ref = blasius_solution()

    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.plot(
        eta_ref,
        fprime_ref,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
        label="Blasius $f'(\\eta)$",
    )

    max_err = 0.0
    for station in sorted(set(data["station"])):
        sel = data["station"] == station
        y, u = data["position_y"][sel], data["velocity_x"][sel]
        sampled_x = float(data["position_x"][sel][0])
        eta = y * np.sqrt(FREESTREAM_SPEED / (kinematic_viscosity * sampled_x))
        u_norm = u / FREESTREAM_SPEED
        marker = STATION_MARKERS.get(station, "d")
        ax.plot(
            eta,
            u_norm,
            marker,
            markersize=3,
            color=COLORS["fvm"],
            linestyle="-",
            label=f"FVM $x/L$ = {sampled_x:.2g}",
        )
        # Error against Blasius inside the layer (eta <= 6).
        inside = eta <= 6.0
        u_ref = np.interp(eta[inside], eta_ref, fprime_ref)
        err = float(np.max(np.abs(u_norm[inside] - u_ref))) if inside.any() else 0.0
        max_err = max(max_err, err)
        print(f"  sampled x/L = {sampled_x:.6g}: max |u/U - f'(eta)| = {err:.4f} (eta <= 6)")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"$\eta = y \sqrt{U_\infty / (\nu x)}$")
    ax.set_ylabel(r"$u / U_\infty$")
    ax.grid(False)
    ax.legend()

    centered_subplots_adjust(fig, outer=0.100, bottom=0.20, top=0.984)
    save_fig(fig, "blasius_profiles.png", FIGURES_DIR, dpi=args.dpi, figure_format=args.format)
    print(
        f"  overall max profile error: {max_err:.4f}"
        f"  [{'OK' if max_err < 0.05 else 'OUT OF BAND'} — target < 0.05]"
    )


if __name__ == "__main__":
    main()
