#!/usr/bin/env python3
"""Rotor performance — thrust_coefficient / power_coefficient time history and actuator-disk theory comparison.

Reads ``samples/rotor/vlm_forces.csv`` and produces two subplots:

1. Thrust coefficient (thrust_coefficient) and power coefficient (power_coefficient) versus time,
   with Betz-limit reference lines.
2. power_coefficient–thrust_coefficient operating trajectory overlaid on the actuator-disk momentum
   theory envelope ``power_coefficient = 0.5·thrust_coefficient·(1 + sqrt(1-thrust_coefficient))``.

Saves: ``figures/rotor_performance.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    build_arg_parser,
    build_rotor_style_map,
    DENSITY,
    FIGURES_DIR,
    FREESTREAM_SPEED,
    load_theme,
    ROTOR_RADIUS,
    SAMPLES_DIR,
    TIP_SPEED_RATIO,
)

# ==============================================================================
# Physics helpers
# ==============================================================================


def actuator_disk_cp(thrust_coefficient: np.ndarray) -> np.ndarray:
    """Actuator-disk momentum theory: power_coefficient as a function of thrust_coefficient."""
    thrust_coefficient = np.asarray(thrust_coefficient)
    power_coefficient = np.zeros_like(thrust_coefficient, dtype=float)
    mask = (thrust_coefficient >= 0) & (thrust_coefficient <= 1.0)
    power_coefficient[mask] = (
        0.5 * thrust_coefficient[mask] * (1.0 + np.sqrt(1.0 - thrust_coefficient[mask]))
    )
    return power_coefficient


# ==============================================================================
# Plot
# ==============================================================================


def plot_rotor_performance(args) -> int:
    fmt = getattr(args, "format", "png")
    out = FIGURES_DIR / f"rotor_performance.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = SAMPLES_DIR / "vlm_forces.csv"
    if not csv_path.exists():
        print(f"[rotor] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[rotor] CSV is empty.")
        return 1

    # Physical constants
    density = DENSITY
    freestream_speed = FREESTREAM_SPEED
    rotor_radius = ROTOR_RADIUS
    angular_velocity = TIP_SPEED_RATIO * freestream_speed / rotor_radius
    rotor_disk_area = np.pi * rotor_radius**2
    dynamic_pressure = 0.5 * density * freestream_speed**2
    dynamic_pressure_area = dynamic_pressure * rotor_disk_area
    dynamic_pressure_area_speed = dynamic_pressure_area * freestream_speed

    # Coefficients
    # freestream_speed is +x and omega_vec is -angular_velocity*xhat.
    thrust_coefficient = df["force_x"].to_numpy() / dynamic_pressure_area
    power_coefficient = (
        -df["moment_x"].to_numpy() * angular_velocity
    ) / dynamic_pressure_area_speed
    time = df["time"].to_numpy()
    rotations = time * angular_velocity / (2.0 * np.pi)

    colors, _ = load_theme()
    styles = build_rotor_style_map(colors)
    s_ref = styles["reference"]
    betz_thrust_coefficient = 8.0 / 9.0
    betz_power_coefficient = 16.0 / 27.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 6.0 / 2.54))
    fig.subplots_adjust(wspace=0.23, top=0.95, bottom=0.13, left=0.14, right=0.96)

    # -- thrust_coefficient & power_coefficient vs time -------------------------------------------------
    ax1 = axes[0]
    thrust_coefficient_style = styles["thrust_coefficient"]
    power_coefficient_style = styles["power_coefficient"]

    ax1.plot(
        rotations,
        thrust_coefficient,
        color=thrust_coefficient_style["color"],
        marker=thrust_coefficient_style["marker"],
        markersize=thrust_coefficient_style["markersize"],
        lw=thrust_coefficient_style["linewidth"],
        label=thrust_coefficient_style["label"],
        markevery=5,
    )
    ax1.plot(
        rotations,
        power_coefficient,
        color=power_coefficient_style["color"],
        marker=power_coefficient_style["marker"],
        markersize=power_coefficient_style["markersize"],
        lw=power_coefficient_style["linewidth"],
        label=power_coefficient_style["label"],
        markevery=5,
    )

    ax1.axhline(
        betz_thrust_coefficient, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"]
    )
    ax1.axhline(
        betz_power_coefficient, color=s_ref["color"], ls=s_ref["linestyle"], lw=s_ref["linewidth"]
    )

    ax1.text(
        7,
        1.01 * betz_thrust_coefficient,
        r"$C_{T,\mathrm{Betz}}$",
        color=s_ref["color"],
        va="bottom",
        ha="left",
    )
    ax1.text(
        7,
        1.01 * betz_power_coefficient,
        r"$C_{P,\mathrm{Betz}}$",
        color=s_ref["color"],
        va="bottom",
        ha="left",
    )

    ax1.set_xlabel(r"Revolutions")
    ax1.set_ylabel(r"Coefficient")
    ax1.set_xlim([0, rotations[-1]])
    ax1.set_ylim([0, 1.1])
    ax1.legend(loc="lower center", ncol=2)
    ax1.set_title(r"Rotor performance coefficients")

    # -- power_coefficient vs thrust_coefficient with theory envelope -----------------------------------
    ax2 = axes[1]
    ct_theory = np.linspace(0.0, 1.0, 300)
    cp_theory = actuator_disk_cp(ct_theory)

    ax2.plot(
        ct_theory,
        cp_theory,
        zorder=0,
        **styles["theory"],
    )

    trajectory_color = styles["vpm"]["color"]
    if len(thrust_coefficient) > 1:
        points = np.column_stack([thrust_coefficient, power_coefficient]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_alpha = np.linspace(0.05, 1.0, len(thrust_coefficient))[1:]
        segment_colors = [to_rgba(trajectory_color, alpha=a) for a in segment_alpha]
        trajectory = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=1.0,
            zorder=1,
        )
        ax2.add_collection(trajectory)
    else:
        segment_alpha = np.array([1.0])

    markevery = max(1, len(thrust_coefficient) // 24)
    marker_idx = np.arange(0, len(thrust_coefficient), markevery)
    if marker_idx[-1] != len(thrust_coefficient) - 1:
        marker_idx = np.append(marker_idx, len(thrust_coefficient) - 1)
    marker_alpha = np.linspace(0.0, 1.0, len(thrust_coefficient))[marker_idx]
    marker_colors = [to_rgba(trajectory_color, alpha=a) for a in marker_alpha]
    ax2.scatter(
        thrust_coefficient[marker_idx],
        power_coefficient[marker_idx],
        color=marker_colors,
        marker="s",
        s=styles["vpm"]["markersize"] ** 2,
        zorder=2,
    )

    ax2.scatter(
        [betz_thrust_coefficient],
        [betz_power_coefficient],
        color=s_ref["color"],
        marker="*",
        s=24,
        zorder=2,
    )

    ax2.set_xlabel(r"$C_T$")
    ax2.set_ylabel(r"$C_P$")
    ax2.set_xlim([0, 1.0])
    ax2.set_ylim(
        [0, max(0.7, float(np.nanmax(power_coefficient)) * 1.08, betz_power_coefficient * 1.08)]
    )

    from matplotlib.lines import Line2D

    ax2.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=trajectory_color,
                lw=1.0,
                marker="s",
                markersize=3,
                label="VLM-VPM time trajectory",
            ),
            Line2D([0], [0], color=s_ref["color"], ls="--", lw=1.0, label="Actuator-disk theory"),
            Line2D([0], [0], color=s_ref["color"], marker="*", ms=6, lw=0, label="Betz limit"),
        ],
        loc="lower center",
    )
    ax2.set_title(r"Operating trajectory")

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    return plot_rotor_performance(
        build_arg_parser(
            "Rotor performance thrust_coefficient/power_coefficient plotting."
        ).parse_args()
    )


if __name__ == "__main__":
    raise SystemExit(main())
