#!/usr/bin/env python3
"""
Spanwise loading validation — VLM bound circulation vs BEM prediction.
======================================================================
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import CM, build_arg_parser, load_theme, rotor_styles, save_figure


def _read_vlm_spanwise(
    samples_dir: Path,
    surface: str,
    tail_fraction: float,
    averaging_rotations: float,
    omega: float,
    hub_r: float,
    tip_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (r, |Gamma|, chord, cl_from_gamma) averaged over the tail.

    The blade rotates, so global coordinates are not stable span coordinates.
    The sampler writes one ordered spanwise row per time snapshot; grouping by
    that row order avoids artificial sawtooth curves as the blade azimuth
    changes.  The CSV ``cl`` column is a fixed wind-axis quantity and is not a
    sectional rotor-blade coefficient, so the plot reconstructs an equivalent
    sectional coefficient from bound circulation.
    """
    csv_path = samples_dir / f"vlm_spanwise_{surface}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    if "half" in df.columns:
        df = df[df["half"] == "orig"]

    t_max = float(df["time"].max())
    if averaging_rotations > 0.0 and omega > 0.0:
        t_cut = max(float(df["time"].min()), t_max - averaging_rotations * 2.0 * np.pi / omega)
    else:
        t_cut = t_max * (1.0 - tail_fraction)
    df = df.sort_values(["time"]).copy()
    df["station"] = df.groupby("time").cumcount()
    tail = df[df["time"] >= t_cut].copy()
    tail["Gamma_abs"] = tail["Gamma"].abs()

    gp = tail.groupby("station")[["Gamma_abs", "chord_local"]].mean().reset_index()
    n_stations = len(gp)
    edges = np.linspace(hub_r, tip_r, n_stations + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    gamma = gp["Gamma_abs"].to_numpy()
    chord = gp["chord_local"].to_numpy()
    return r, gamma, chord, tail["time"].to_numpy()


def main() -> int:
    ap = build_arg_parser("Spanwise loading validation: VLM vs BEM")
    ap.add_argument(
        "--tail-fraction",
        type=float,
        default=0.25,
        help="Fraction of simulation tail used for time-averaging (default 0.25 = last 25%%).",
    )
    ap.add_argument(
        "--averaging-rotations",
        type=float,
        default=3.0,
        help="Number of final rotor rotations to average (default: 3).",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(SCRIPT_DIR))
    from rotor_theory import bem_solve
    from generate_openvsp_blade import RotorBladeDesign, design_schedule

    samples = Path(args.solution_dir) / "samples"
    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # ── BEM reference ────────────────────────────────────────────────────────
    design = RotorBladeDesign()
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

    # ── VLM time-averaged spanwise data ──────────────────────────────────────
    result = _read_vlm_spanwise(
        samples,
        surface="blade_0",
        tail_fraction=args.tail_fraction,
        averaging_rotations=args.averaging_rotations,
        omega=design.omega,
        hub_r=design.hub_radius,
        tip_r=design.radius,
    )
    if result is None:
        print(
            f"  [WARNING] vlm_spanwise_blade_0.csv not found in {samples} — "
            "run rotorFlow first."
        )
        return 0

    r_vlm, gamma_vlm, chord_vlm, _ = result
    vrel_geom = np.sqrt(design.freestream_velocity**2 + (design.omega * r_vlm) ** 2)
    cl_vlm = 2.0 * gamma_vlm / (chord_vlm * vrel_geom)

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors, _ = load_theme()
    styles = rotor_styles(colors)
    color_vlm = styles["vpm"]["color"]
    color_bem = styles["bem"]["color"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8 * CM, 7.4 * CM), sharex=True)
    fig.subplots_adjust(wspace=0.18, left=0.12, right=0.96, top=0.87, bottom=0.30)

    # ── Γ(r/R) ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(
        r_vlm / design.radius,
        gamma_vlm,
        color=color_vlm,
        marker=styles["vpm"]["marker"],
        markersize=styles["vpm"]["markersize"],
        lw=styles["vpm"]["linewidth"],
        label=r"VLM-VPM",
    )
    ax.plot(
        bem["r_over_R"],
        bem["Gamma"],
        color=color_bem,
        ls=styles["bem"]["linestyle"],
        lw=styles["bem"]["linewidth"],
        label=r"BEM reference",
    )
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$\Gamma$ [m$^2$/s]")
    ax.set_title(r"Bound circulation")
    ax.set_xlim([0, 1])
    gamma_top = max(float(np.nanmax(gamma_vlm)), float(np.nanmax(bem["Gamma"]))) * 1.08
    ax.set_ylim([0, gamma_top])

    # ── Cl(r/R) ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(
        r_vlm / design.radius,
        cl_vlm,
        color=color_vlm,
        marker=styles["vpm"]["marker"],
        markersize=styles["vpm"]["markersize"],
        lw=styles["vpm"]["linewidth"],
        label=r"VLM-VPM",
    )
    ax2.plot(
        bem["r_over_R"],
        bem["Cl"],
        color=color_bem,
        ls=styles["bem"]["linestyle"],
        lw=styles["bem"]["linewidth"],
        label=r"BEM reference",
    )
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

    out = figs / f"rotor_loading_validation.{args.format}"
    save_figure(fig, out, args.dpi, args.format)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(
        f"  BEM Ct={bem_ct:.4f}, Cp={bem_cp:.4f}"
        f" (Betz: Ct={8/9:.4f}, Cp={16/27:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
