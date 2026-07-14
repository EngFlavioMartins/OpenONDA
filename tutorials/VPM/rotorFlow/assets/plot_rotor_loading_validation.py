#!/usr/bin/env python3
"""Spanwise loading validation — VLM bound circulation vs BEM prediction.

Reads ``solution/rotor/samples/vlm_spanwise_blade_0.csv`` produced by the
loading-distribution sampler and compares the time-averaged bound circulation
Γ(r/R) against the BEM prediction for the same Betz-optimal blade geometry
(TSR = 7, three blades, Selig–Betz twist from hub to tip).

A second panel shows the spanwise sectional lift coefficient Cl(r/R), which
provides an independent check that the VLM is operating at the design angle of
attack (~5° once the wake induction has fully developed).

Saves: ``figures/rotor_loading_validation.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_arg_parser, build_rotor_style_map, load_theme

# ==============================================================================
# Data loader
# ==============================================================================


def _read_vlm_spanwise(
    samples_dir: Path,
    surface: str,
    tail_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (r, |Gamma|, chord, Cl, last_times) averaged over the tail.

    Reads the per-station CSV, filters to the last ``tail_fraction`` of the
    simulation timeline, and averages ``Gamma_abs`` and ``cl_from_gamma`` over
    that window per physical span station.  ``r`` and ``chord_local`` are taken
    from the first occurrence of each station (they are constant in time).
    """
    csv_path = samples_dir / f"vlm_spanwise_{surface}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    t_max = float(df["time"].max())
    t_cut = t_max * (1.0 - tail_fraction)
    steady = df[df["time"] >= t_cut].copy()

    gp = (
        steady.groupby("station_id", sort=False)
        .agg(
            r=("span_coordinate_abs", "first"),
            Gamma=("Gamma_abs", "mean"),
            chord=("chord_local", "first"),
            cl=("cl_from_gamma", "mean"),
        )
        .sort_values("r")
        .reset_index(drop=True)
    )

    return (
        gp["r"].to_numpy(),
        gp["Gamma"].to_numpy(),
        gp["chord"].to_numpy(),
        gp["cl"].to_numpy(),
        steady["time"].to_numpy(),
    )


# ==============================================================================
# Plot
# ==============================================================================


def plot_loading_validation(args) -> int:
    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    fmt = getattr(args, "format", "png")
    out = figs / f"rotor_loading_validation.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)

    # -- BEM reference ---------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent))
    from rotor_theory import bem_solve
    from generate_openvsp_blade import RotorBladeDesign, design_schedule

    design = RotorBladeDesign(n_stations=23, chord_stations=7)
    sched = design_schedule(design)
    bem = bem_solve(
        r=sched["r"],
        chord=sched["chord"],
        twist_rad=np.radians(sched["theta_deg"]),
        B=3,
        R=design.radius,
        U_inf=design.freestream_velocity,
        omega=design.omega,
    )
    bem_ct = bem.attrs["Ct"]
    bem_cp = bem.attrs["Cp"]

    # -- VLM time-averaged spanwise data ---------------------------------
    result = _read_vlm_spanwise(
        samples,
        surface="blade_0",
        tail_fraction=args.tail_fraction,
    )
    if result is None:
        print(f"  [WARNING] vlm_spanwise_blade_0.csv not found in {samples} — run rotorFlow first.")
        return 0

    r_vlm, gamma_vlm, chord_vlm, cl_vlm, _ = result

    # Normalised circulation reference: Γ* = Γ / (U∞ R)
    gamma_ref = design.freestream_velocity * design.radius
    gamma_vlm_norm = gamma_vlm / gamma_ref
    bem_gamma_norm = bem["Gamma"].to_numpy() / gamma_ref

    colors, _ = load_theme()
    styles = build_rotor_style_map(colors)
    s_vpm = styles["vpm"]
    s_bem = styles["bem"]
    s_ref = styles["reference"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8 / 2.54, 7.4 / 2.54), sharex=True)
    fig.subplots_adjust(wspace=0.27, left=0.12, right=0.96, top=0.87, bottom=0.27)

    # -- Gamma*(r/R) -----------------------------------------------------
    ax = axes[0]
    ax.axvspan(0, 0.17, color=colors["background_light"], linewidth=0)
    vpm_kw = {
        "color": s_vpm["color"],
        "marker": s_vpm["marker"],
        "markersize": s_vpm["markersize"],
        "lw": s_vpm["linewidth"],
        "label": s_vpm["label"],
    }
    bem_kw = {
        "color": s_bem["color"],
        "ls": s_bem["linestyle"],
        "lw": s_bem["linewidth"],
        "label": s_bem["label"],
    }
    ax.plot(r_vlm / design.radius, gamma_vlm_norm, **vpm_kw)
    ax.plot(bem["r_over_R"], bem_gamma_norm, **bem_kw)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\Gamma\,/\,(U_\infty R)$")
    ax.set_title(r"Normalized bound circulation")
    ax.set_xlim([0, 1])
    gamma_top = max(float(np.nanmax(gamma_vlm_norm)), float(np.nanmax(bem_gamma_norm))) * 1.08
    ax.set_ylim([0, gamma_top])

    # -- Cl(r/R) ---------------------------------------------------------
    ax2 = axes[1]
    ax2.axvspan(0, 0.17, color=colors["background_light"], linewidth=0)
    ax2.plot(r_vlm / design.radius, cl_vlm, **vpm_kw)
    ax2.plot(bem["r_over_R"], bem["Cl"], **bem_kw)
    ax2.set_xlabel(r"$r/R$")
    ax2.set_ylabel(r"$C_l$")
    ax2.set_title(r"Sectional lift coefficient")
    ax2.set_xlim([0, 1])
    cl_top = max(float(np.nanmax(cl_vlm)), float(np.nanmax(bem["Cl"]))) * 1.08
    ax2.set_ylim([0, cl_top])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
        columnspacing=1.1,
        handlelength=2.2,
    )

    save_kw: dict = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = args.dpi
    plt.savefig(out, **save_kw)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  BEM Ct={bem_ct:.4f}, Cp={bem_cp:.4f} (Betz: Ct={8 / 9:.4f}, Cp={16 / 27:.4f})")
    return 0


def main() -> int:
    p = build_arg_parser("Spanwise loading validation: VLM vs BEM")
    p.add_argument(
        "--tail-fraction",
        type=float,
        default=0.30,
        help="Fraction of simulation tail used for time-averaging (default 0.30 = last 30%%).",
    )
    return plot_loading_validation(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
