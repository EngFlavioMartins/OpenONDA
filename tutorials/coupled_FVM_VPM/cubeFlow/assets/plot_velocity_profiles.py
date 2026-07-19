#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from _plotutil import CASE_DIR, COLORS, load_forces, run_constants, save
from _reference_util import (
    MATCH_TOL,
    assert_same_time,
    _pvd_times,
    load_vpm_particles,
    nearest_vpm_h5,
    nearest_vtu,
    sample_vtu,
)
from _frames_util import D, EPS, U_INF, vpm_velocity

FIG = CASE_DIR / "figures"
REF_PVD = CASE_DIR / "referenceFlow" / "solution" / "referenceFlow.pvd"
REF_FORCES = CASE_DIR / "referenceFlow" / "solution" / "forces_history.csv"


def fig_velocity_profiles(t, hyb_vtu, ref_s, particles, box, forces, fmt, dpi):
    x = np.linspace(-3.0, 10.0, 521)
    cl = np.column_stack([x, np.full_like(x, EPS), np.full_like(x, EPS)])
    oa = np.column_stack([x, np.full_like(x, 0.75), np.full_like(x, EPS)])

    in_box = (x >= box["xmin"]) & (x <= box["xmax"])
    hyb_cl = sample_vtu(hyb_vtu, cl)["U"][:, 0]
    hyb_oa = sample_vtu(hyb_vtu, oa)["U"][:, 0]
    ref_cl = sample_vtu(ref_s[1], cl)["U"][:, 0]
    ref_oa = sample_vtu(ref_s[1], oa)["U"][:, 0]
    vpm_dom = (x >= -2.0) & (x <= 10.0)
    vpm_cl = np.full_like(x, np.nan)
    vpm_oa = np.full_like(x, np.nan)
    vpm_cl[vpm_dom] = vpm_velocity(particles, cl[vpm_dom])[:, 0]
    vpm_oa[vpm_dom] = vpm_velocity(particles, oa[vpm_dom])[:, 0]

    fig_height = 6.0 / 2.54
    fig_length = 12.5 / 2.54
    fig = plt.figure(figsize=(fig_length, 2 * fig_height), dpi=400)
    gs = GridSpec(2, 2, figure=fig)
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, :])

    t_h, cd_h = forces["hyb"]
    t_r, cd_r = forces["ref"]
    if t_h.size > 0 and np.any(cd_h != 0):
        ax1.plot(
            t_h * U_INF / D,
            cd_h,
            color=COLORS["hybrid"],
            ls="-",
            marker="o",
            ms=2,
            label="FVM",
        )
    if t_r.size > 0 and np.any(cd_r != 0):
        ax1.plot(t_r * U_INF / D, cd_r, color=COLORS["reference"], ls="-.", label="Reference FVM")
    ax1.set(
        xlabel=r"$t U_\infty / D$",
        ylabel=r"$C_D$",
        xlim=(0, 20),
        ylim=(0.5, 2),
        title=r"Drag Coefficient, $C_D$",
    )
    ax1.legend(loc="upper right")

    for ax, hyb, ref, vpm, ttl, ylim in (
        (ax2, hyb_cl, ref_cl, vpm_cl, f"Centerline ($t={t:.2f}$)", (-1.2, 1.2)),
        (ax3, hyb_oa, ref_oa, vpm_oa, f"Off-axis $y=0.75D$ ($t={t:.2f}$)", (-0.5, 1.5)),
    ):
        ax.axvspan(box["xmin"], box["xmax"], color=COLORS["background_light"], zorder=0)
        if ax is ax2:
            ax.axvspan(-0.5, 0.5, color=COLORS["background_strong"], zorder=1)
        ax.plot(x, ref / U_INF, color=COLORS["reference"], ls="-.", label="Reference FVM", zorder=2)
        hx = np.where(in_box, hyb, np.nan)
        ax.plot(x, hx / U_INF, color=COLORS["hybrid"], ls="-", label="FVM", zorder=2)
        ax.plot(
            x,
            vpm / U_INF,
            color=COLORS["vpm"],
            ls="-",
            marker="o",
            ms=1.5,
            markevery=5,
            label="VPM",
            zorder=2,
        )
        ax.set(xlabel=r"$x/D$", ylabel="", xlim=(-3, 10), ylim=ylim, title=ttl)
    ax2.set_ylabel(r"$u_x/U_\infty$")
    ax3.legend(loc="lower right")

    save(fig, f"velocity_profiles_t{t:.2f}", fmt, dpi or 400)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default="all", help="all | latest | comma list")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    FIG.mkdir(exist_ok=True)
    C = run_constants()
    box = C["box"]

    hyb_pvd = next(iter((CASE_DIR / "solution").glob("coupled_*.pvd")), None)
    if hyb_pvd is None:
        sys.exit("no hybrid .pvd in solution/")
    entries = [(tt, f) for tt, f in _pvd_times(hyb_pvd) if tt > 1e-9]
    if args.times == "latest":
        entries = entries[-1:]
    elif args.times != "all":
        wanted = [float(v) for v in args.times.split(",")]
        entries = [min(entries, key=lambda e: abs(e[0] - w)) for w in wanted]

    hf = load_forces()
    forces = {"hyb": (np.asarray(hf.get("time", [])), np.asarray(hf.get("Cd", [])))}
    if REF_FORCES.exists():
        rows = np.atleast_1d(
            np.genfromtxt(REF_FORCES, delimiter=",", names=True, dtype=None, encoding="utf-8")
        )
        forces["ref"] = (rows["time"], rows["Cd"])
    else:
        forces["ref"] = (np.array([]), np.array([]))

    for t, hyb_vtu in entries:
        out = FIG / f"velocity_profiles_t{t:.2f}.{args.format}"
        if not args.force and out.exists():
            continue
        h5 = nearest_vpm_h5(t)
        particles = (
            load_vpm_particles(h5)
            if h5
            else {
                "position": np.zeros((0, 3)),
                "circulation": np.zeros((0, 3)),
                "radius": np.zeros(0),
            }
        )
        ref_s = nearest_vtu(REF_PVD, t)
        if ref_s is None or abs(ref_s[0] - t) > MATCH_TOL:
            print(f"  velocity_profiles t={t:.2f}: no coincident reference snapshot — skip")
            continue
        assert_same_time(t, ref_s[0])
        fig_velocity_profiles(t, hyb_vtu, ref_s, particles, box, forces, args.format, args.dpi)
        print(f"  velocity_profiles t={t:.2f} (ref t={ref_s[0]:.2f}) done")


if __name__ == "__main__":
    main()
