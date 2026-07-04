#!/usr/bin/env python3
"""Energy dissipation — normalised power across all three test cases.

Reads flow-integral CSV files (``samples/flow_integrals.csv``) exported by
the VPM solver when logging is active.

Plots a 3x1 figure:
  - Top panel:    single vortex
  - Centre panel: vortex dipole
  - Bottom panel: co-rotating merger

Filled markers show dE/dt (energy decay rate), hollow markers show -νΩ
(viscous dissipation via enstrophy).  Colours are consistent per scheme
across all three panels.

Saves: figures/lamboseen_energy.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    SCHEMES,
    add_physics_args,
    build_arg_parser,
    build_style_map,
    load_theme,
    resolve_runtime_physics,
)


# =============================================================
# CSV reader
# =============================================================


def read_flow_integrals(csv_path: Path) -> dict | None:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "dEdt" not in df.columns:
        return None
    return {
        "t": df["time"].to_numpy(),
        "dedt": df["dEdt"].to_numpy(),
        "nu_omega": df["neg_nu_enstrophy"].to_numpy(),
    }


# =============================================================
# Plot
# =============================================================

CASES = (
    ("vortex", "Single vortex", 1),
    ("dipole", "Vortex dipole", 2),
    ("merging", "Co-rotating merger", 2),
)


def prepend_initial_point(data: dict, gamma: float, t0: float, n_vortices: int) -> dict:
    if len(data["t"]) == 0 or data["t"][0] == 0.0:
        return data
    initial_power = -n_vortices * gamma**2 / (8.0 * np.pi * t0)
    return {
        "t": np.insert(data["t"], 0, 0.0),
        "dedt": np.insert(data["dedt"], 0, initial_power),
        "nu_omega": np.insert(data["nu_omega"], 0, initial_power),
    }


def plot_case_panel(
    ax,
    solution_dir: Path,
    case_prefix: str,
    title: str,
    n_vortices: int,
    style_map: dict,
    tau_scale: float,
    p_ref: float,
    gamma: float,
    t0: float,
) -> None:
    ax.set_title(title)
    for scheme in SCHEMES:
        csv_path = solution_dir / f"{case_prefix}_{scheme}" / "samples" / "flow_integrals.csv"
        data = read_flow_integrals(csv_path)
        if data is None:
            continue
        data = prepend_initial_point(data, gamma, t0, n_vortices)
        st = style_map[scheme]
        tau = data["t"] * tau_scale
        plot_kw = {
            "color": st["color"],
            "marker": st["marker"],
            "markersize": 2.2,
            "linestyle": "None",
            "linewidth": 1.0,
            "alpha": 0.85,
            "markevery": 4,
        }
        ax.plot(tau, data["dedt"] / p_ref, **plot_kw)
        ax.plot(tau, data["nu_omega"] / p_ref, mfc="none", **plot_kw)


def plot_energy_enstrophy(args) -> int:
    solution_dir = Path(args.solution_dir)
    fmt = getattr(args, "format", "png")
    out = Path(args.figures_dir) / f"lamboseen_energy.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    colors, theme = load_theme()
    if theme is not None and hasattr(theme, "set_style"):
        theme.set_style()
    style_map = build_style_map(colors)
    runtime = resolve_runtime_physics(solution_dir, args.gamma, args.nu, args.b0, args.a0_over_b0)
    run_nu = runtime["nu"]
    a0 = runtime["ac0"]
    run_t0 = runtime["t0"]

    tau_scale = run_nu / (a0**2)
    p_ref = run_nu * (args.gamma**2) / (a0**2)
    tau_lim = 0.04 / (args.a0_over_b0**2)

    fig, axes = plt.subplots(1, 3, figsize=(12.8 / 2.54, 7.5 / 2.54), sharey='row')
    fig.subplots_adjust(wspace=0.10, top=0.96, bottom=0.35, left=0.11, right=0.89)

    for ax, (case_prefix, title, n_vortices) in zip(axes, CASES):
        plot_case_panel(
            ax,
            solution_dir,
            case_prefix,
            title,
            n_vortices,
            style_map,
            tau_scale,
            p_ref,
            args.gamma,
            run_t0,
        )
        
        ax.set_ylim([-0.5, 0])
        ax.set_xlim([0, tau_lim])
        ax.set_xlabel(r"$\nu t / a_0^2$")

    axes[0].set_ylabel(r"$(dE/dt) / (\nu\Gamma^2 / a_0^2)$")

    handles: list = []
    for scheme in SCHEMES:
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
        bbox_to_anchor=(0.5, 0.02) ,
    )

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    return 0


def main() -> int:
    p = build_arg_parser("Energy balance: dE/dt")
    add_physics_args(p)
    return plot_energy_enstrophy(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
