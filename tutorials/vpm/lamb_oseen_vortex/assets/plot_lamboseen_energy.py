#!/usr/bin/env python3
"""Energy dissipation — normalised power across all three test cases.

Reads flow-integral CSV files (``samples/flow_integrals.csv``) exported by
the VPM solver when logging is active.

Plots one panel for each physics case: single vortex, vortex dipole, and
co-rotating merger.

Filled markers show dE/dt (energy decay rate), hollow markers show -νΩ
(viscous dissipation via enstrophy).  Colours are consistent per scheme
across all three panels.

Saves: figures/lamboseen_energy.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __package__:
    from .plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from .vortex_diagnostics import SCHEMES, resolve_runtime_physics
else:
    from plot_style import build_arg_parser, build_style_map, figure_size, load_theme, save_fig
    from vortex_diagnostics import SCHEMES, resolve_runtime_physics


# =============================================================
# CSV reader
# =============================================================


def read_flow_integrals(csv_path: Path) -> dict | None:
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip").dropna(subset=["time"])
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"  [energy] skipping unreadable live CSV {csv_path.name}: {exc}")
        return None
    if df.empty or "kinetic_energy_rate" not in df.columns:
        return None
    return {
        "time": df["time"].to_numpy(),
        "kinetic_energy_rate": df["kinetic_energy_rate"].to_numpy(),
        "viscous_kinetic_energy_rate": df["viscous_kinetic_energy_rate"].to_numpy(),
    }


# =============================================================
# Plot
# =============================================================

CASES = (
    ("vortex", "Single vortex", 1),
    ("dipole", "Vortex dipole", 2),
    ("merging", "Co-rotating merger", 2),
)


def prepend_initial_point(
    data: dict,
    circulation: float,
    t0: float,
    n_vortices: int,
    column_length: float,
) -> dict:
    if len(data["time"]) == 0 or data["time"][0] == 0.0:
        return data
    initial_power = -n_vortices * circulation**2 * column_length / (8.0 * np.pi * t0)
    return {
        "time": np.insert(data["time"], 0, 0.0),
        "kinetic_energy_rate": np.insert(data["kinetic_energy_rate"], 0, initial_power),
        "viscous_kinetic_energy_rate": np.insert(
            data["viscous_kinetic_energy_rate"], 0, initial_power
        ),
    }


def plot_case_panel(
    ax,
    samples_dir: Path,
    case_prefix: str,
    title: str,
    n_vortices: int,
    style_map: dict,
    tau_scale: float,
    p_ref: float,
    circulation: float,
    t0: float,
    column_length: float,
) -> float:
    ax.set_title(title)
    latest_tau = 0.0
    for scheme in SCHEMES:
        csv_path = samples_dir / f"{case_prefix}_{scheme}" / "flow_integrals.csv"
        data = read_flow_integrals(csv_path)
        if data is None:
            continue
        data = prepend_initial_point(data, circulation, t0, n_vortices, column_length)
        st = style_map[scheme]
        tau = data["time"] * tau_scale
        plot_kw = {
            "color": st["color"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
            "alpha": 0.85,
        }
        ax.plot(tau, data["kinetic_energy_rate"] / p_ref, zorder=10, **plot_kw)
        ax.plot(
            tau,
            data["viscous_kinetic_energy_rate"] / p_ref,
            mfc="none",
            zorder=100,
            **plot_kw,
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
    fig, axes = plt.subplots(1, 3, figsize=figure_size("trajectory"), sharey=True)
    fig.subplots_adjust(wspace=0.09, top=0.92, bottom=0.32, left=0.12, right=0.98)

    plotted = False
    for ax, (case_prefix, title, n_vortices) in zip(axes, CASES):
        latest_tau = plot_case_panel(
            ax,
            samples_dir,
            case_prefix,
            title,
            n_vortices,
            style_map,
            tau_scale,
            p_ref,
            run_circulation,
            run_t0,
            column_length,
        )
        plotted |= latest_tau > 0.0

        ax.set_xlabel(r"$\nu t / a_{c,0}^2$")

    if not plotted:
        plt.close(fig)
        out.unlink(missing_ok=True)
        print("  [energy] no sampled flow integrals; figure not generated")
        return 0

    axes[0].set_ylabel(r"$(dE/dt) / (\nu\Gamma^2 L / a_{c,0}^2)$")

    axes[0].set_xlim([0.0, 3.8])
    axes[1].set_xlim([0.0, 3.8])
    axes[2].set_xlim([0.0, 3.8])

    # sharey=True links the y-axes, so the limit is set once and propagates.
    axes[0].set_ylim([-1.5, 0.2])

    available_schemes = [
        scheme
        for scheme in SCHEMES
        if any(
            read_flow_integrals(samples_dir / f"{case_prefix}_{scheme}" / "flow_integrals.csv")
            is not None
            for case_prefix, _, _ in CASES
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
                linestyle="None",
                marker=st["marker"],
                markersize=4,
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
            linestyle="None",
            marker="o",
            markersize=4,
            mfc="none",
            label=r"$-\nu\Omega$",
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
